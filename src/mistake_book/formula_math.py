from __future__ import annotations

import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass

from defusedxml import ElementTree as DefusedElementTree
from latex2mathml.converter import convert as latex_to_mathml_text
import math_ml2omml


MAX_LATEX_BYTES = 8192
MAX_LATEX_TOKENS = 2048
MAX_NESTING = 32

_COMMAND = re.compile(r"\\([A-Za-z]+|.)")
_ENVIRONMENT = re.compile(r"\\(?:begin|end)\s*\{([A-Za-z*]+)\}")
_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|\{|\}|\[|\]|[^\s]")
_SAFE_COMMANDS = {
    "acute", "aleph", "alpha", "angle", "approx", "arccos", "arcsin",
    "arctan", "arg", "array", "ast", "bar", "begin", "beta", "bf",
    "big", "Big", "bigg", "Bigg", "bigl", "Bigl", "biggl", "Biggl",
    "bigr", "Bigr", "biggr", "Biggr", "binom", "bmatrix", "boldsymbol",
    "bot", "boxed", "breve", "bullet",
    "cases", "cdot", "cdots", "chi", "circ", "cos", "cosh", "cot",
    "coth", "csc", "ddot", "ddots", "degree", "delta", "Delta", "det",
    "displaystyle", "div", "dot", "dots", "downarrow", "ell", "end",
    "epsilon", "equiv", "eta", "exists", "exp", "fallingdotseq", "forall",
    "frac", "Gamma", "gcd", "ge", "geq", "gets", "gg", "hat", "hbar",
    "hookleftarrow", "hookrightarrow", "iff", "imath", "in", "infty",
    "int", "iota", "jmath", "kappa", "lambda", "Lambda", "langle", "lceil",
    "ldots", "le", "left", "leftarrow", "leftrightarrow", "leq", "lfloor",
    "lg", "lim", "liminf", "limsup", "ll", "ln", "log", "longleftarrow",
    "longleftrightarrow", "longrightarrow", "mapsto", "mathbb", "mathbf",
    "mathcal", "mathfrak", "mathit", "mathrm", "mathsf", "mathtt", "matrix",
    "max", "min", "mod", "mp", "mu", "nabla", "ne", "neq", "ni", "not",
    "notin", "nu", "odot", "oint", "omega", "Omega", "operatorname",
    "oplus", "otimes", "overbrace", "overline", "partial", "perp", "phi",
    "Phi", "pi", "Pi", "pm", "pmatrix", "Pr", "prod", "propto", "psi",
    "Psi", "qquad", "quad", "rangle", "rceil", "Re", "rfloor", "rho",
    "right", "rightarrow", "risingdotseq", "rm", "sec", "sigma", "Sigma",
    "sin", "sinh", "sqrt", "subset", "subseteq", "sum", "sup", "supset",
    "supseteq", "tan", "tanh", "tau", "text", "theta", "Theta", "times",
    "to", "top", "triangle", "underbrace", "underline", "uparrow",
    "upsilon", "Upsilon", "varepsilon", "varphi", "varpi", "varrho",
    "varsigma", "vartheta", "vdots", "vec", "vee", "vmatrix", "Vmatrix",
    "wedge", "xi", "Xi", "zeta",
}
_SAFE_ENVIRONMENTS = {
    "aligned",
    "array",
    "bmatrix",
    "cases",
    "gathered",
    "matrix",
    "pmatrix",
    "smallmatrix",
    "vmatrix",
    "Vmatrix",
}
_SAFE_MATHML_TAGS = {
    "math", "mrow", "mi", "mn", "mo", "mtext", "mspace", "ms", "mglyph",
    "mfrac", "msqrt", "mroot", "mstyle", "merror", "mpadded", "mphantom",
    "mfenced", "menclose", "msub", "msup", "msubsup", "munder", "mover",
    "munderover", "mmultiscripts", "mtable", "mtr", "mtd", "mlabeledtr",
    "maligngroup", "malignmark", "semantics", "annotation",
}
_SAFE_MATHML_ATTRIBUTES = {
    "accent", "accentunder", "align", "columnalign", "columnlines",
    "columnspacing", "columnspan", "denomalign", "depth", "display",
    "displaystyle", "encoding", "fence", "form", "frame", "height",
    "linethickness", "lspace", "mathbackground", "mathcolor", "mathsize",
    "mathvariant", "maxsize", "minsize", "movablelimits", "notation",
    "numalign", "rowalign", "rowlines", "rowspacing", "rowspan", "rspace",
    "scriptlevel", "separator", "stretchy", "symmetric", "voffset", "width",
}


class FormulaValidationError(ValueError):
    pass


@dataclass(frozen=True)
class FormulaConversion:
    latex: str
    mathml: str


def normalize_latex(value: str) -> str:
    text = value.strip()
    if text.startswith("$$") and text.endswith("$$") and len(text) >= 4:
        text = text[2:-2].strip()
    elif text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([_^])\s*", r"\1", text)
    text = re.sub(r"\\([A-Za-z]+)\s+\{", r"\\\1{", text)
    text = re.sub(r"\{\s+", "{", text)
    text = re.sub(r"\s+\}", "}", text)
    text = re.sub(r"\}\s+\{", "}{", text)
    return text


def _validate_source(latex: str) -> None:
    encoded = latex.encode("utf-8")
    if not latex or len(encoded) > MAX_LATEX_BYTES:
        raise FormulaValidationError("LaTeX is empty or exceeds 8192 bytes")
    tokens = _TOKEN.findall(latex)
    if len(tokens) > MAX_LATEX_TOKENS:
        raise FormulaValidationError("LaTeX exceeds 2048 tokens")
    depth = 0
    maximum = 0
    for token in tokens:
        if token == "{":
            depth += 1
            maximum = max(maximum, depth)
        elif token == "}":
            depth -= 1
            if depth < 0:
                raise FormulaValidationError("LaTeX braces are unbalanced")
    if depth or maximum > MAX_NESTING:
        raise FormulaValidationError("LaTeX nesting is invalid or exceeds 32")
    environments = _ENVIRONMENT.findall(latex)
    if any(item not in _SAFE_ENVIRONMENTS for item in environments):
        raise FormulaValidationError("LaTeX environment is not allowed")
    for command in _COMMAND.findall(latex):
        if command.isalpha() and command not in _SAFE_COMMANDS:
            raise FormulaValidationError(f"LaTeX command is not allowed: \\{command}")


def _local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1]


def _sanitize_mathml(mathml: str) -> str:
    try:
        root = DefusedElementTree.fromstring(mathml)
    except Exception as error:
        raise FormulaValidationError(f"MathML parsing failed: {error}") from error
    for element in root.iter():
        tag = _local_name(element.tag)
        if tag not in _SAFE_MATHML_TAGS:
            raise FormulaValidationError(f"MathML element is not allowed: {tag}")
        element.tag = tag
        for attribute in element.attrib:
            if _local_name(attribute) not in _SAFE_MATHML_ATTRIBUTES:
                raise FormulaValidationError(
                    f"MathML attribute is not allowed: {_local_name(attribute)}"
                )
        element.attrib = {
            _local_name(key): value for key, value in element.attrib.items()
        }
    return ElementTree.tostring(root, encoding="unicode")


def convert_latex(latex: str) -> FormulaConversion:
    normalized = normalize_latex(latex)
    _validate_source(normalized)
    try:
        mathml = latex_to_mathml_text(normalized)
    except Exception as error:
        raise FormulaValidationError(f"LaTeX conversion failed: {error}") from error
    return FormulaConversion(
        latex=normalized,
        mathml=_sanitize_mathml(mathml),
    )


def latex_to_omml(latex: str) -> str:
    converted = convert_latex(latex)
    try:
        omml = math_ml2omml.convert(converted.mathml)
        DefusedElementTree.fromstring(
            '<root xmlns:m="http://schemas.openxmlformats.org/officeDocument/'
            f'2006/math">{omml}</root>'
        )
    except Exception as error:
        raise FormulaValidationError(f"OMML conversion failed: {error}") from error
    return omml
