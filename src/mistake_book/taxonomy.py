from __future__ import annotations

import hashlib
import json
import os
import threading
import unicodedata
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from .classification import TAXONOMY

try:
    import fcntl
except ImportError:  # pragma: no cover - only relevant off POSIX
    fcntl = None  # type: ignore[assignment]


UNCATEGORIZED = "\u672a\u5206\u7c7b"
MAX_GROUPS = 50
MAX_CATEGORIES = 100


class TaxonomyError(ValueError):
    pass


class TaxonomyConflict(TaxonomyError):
    pass


def _builtin_id(kind: str, *names: str) -> str:
    digest = hashlib.sha256("\x1f".join(names).encode("utf-8")).hexdigest()[:16]
    return f"builtin-{kind}-{digest}"


def _normalized_name(value: Any, maximum: int) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not text or len(text) > maximum:
        raise TaxonomyError(f"\u5206\u7c7b\u540d\u79f0\u5fc5\u987b\u4e3a 1 \u81f3 {maximum} \u4e2a\u5b57\u7b26")
    if any(ord(character) < 32 or character in "/\\" for character in text):
        raise TaxonomyError("\u5206\u7c7b\u540d\u79f0\u4e0d\u80fd\u5305\u542b\u63a7\u5236\u5b57\u7b26\u6216\u8def\u5f84\u5206\u9694\u7b26")
    if text == UNCATEGORIZED:
        raise TaxonomyError("\u201c\u672a\u5206\u7c7b\u201d\u662f\u7cfb\u7edf\u4fdd\u7559\u540d\u79f0")
    return text


def _seed_groups(
    historical_pairs: Iterable[tuple[str, str]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    known: set[tuple[str, str]] = set()
    by_name: dict[str, dict[str, Any]] = {}
    for group_name, categories in TAXONOMY.items():
        if group_name == UNCATEGORIZED:
            continue
        group = {
            "id": _builtin_id("group", group_name),
            "name": group_name,
            "source": "builtin",
            "active": True,
            "categories": [],
        }
        for category_name in categories:
            group["categories"].append(
                {
                    "id": _builtin_id("category", group_name, category_name),
                    "name": category_name,
                    "source": "builtin",
                    "active": True,
                }
            )
            known.add((group_name, category_name))
        groups.append(group)
        by_name[group_name] = group

    for raw_group, raw_category in historical_pairs:
        if not raw_group or not raw_category:
            continue
        if (raw_group, raw_category) in known or (
            raw_group == UNCATEGORIZED and raw_category == UNCATEGORIZED
        ):
            continue
        group = by_name.get(raw_group)
        if group is None:
            group = {
                "id": f"historical-group-{uuid.uuid4().hex}",
                "name": raw_group,
                "source": "historical",
                "active": False,
                "categories": [],
            }
            groups.append(group)
            by_name[raw_group] = group
        if not any(item["name"] == raw_category for item in group["categories"]):
            group["categories"].append(
                {
                    "id": f"historical-category-{uuid.uuid4().hex}",
                    "name": raw_category,
                    "source": "historical",
                    "active": False,
                }
            )
    return groups


class TaxonomyService:
    def __init__(
        self,
        data_dir: Path,
        historical_pairs: Iterable[tuple[str, str]] = (),
    ) -> None:
        self.path = data_dir / "taxonomy.json"
        self.backup_path = data_dir / "taxonomy.json.bak"
        self.lock_path = data_dir / ".taxonomy.lock"
        self._mutex = threading.RLock()
        self.degraded_error: str | None = None
        pairs = list(historical_pairs)
        self._data = self._load(pairs)
        self._reconcile_historical(pairs)

    def _load(self, historical_pairs: list[tuple[str, str]]) -> dict[str, Any]:
        if not self.path.exists():
            data = {"version": 1, "revision": 1, "groups": _seed_groups(historical_pairs)}
            self._validate(data)
            self._write(data, create_backup=False)
            return data
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._validate(data)
            return data
        except (OSError, json.JSONDecodeError, TaxonomyError) as error:
            try:
                backup = json.loads(self.backup_path.read_text(encoding="utf-8"))
                self._validate(backup)
                self._write(backup, create_backup=False)
                return backup
            except (OSError, json.JSONDecodeError, TaxonomyError):
                self.degraded_error = f"{self.path}: {error}"
                return {"version": 1, "revision": 0, "groups": _seed_groups(())}

    @staticmethod
    def _validate(data: dict[str, Any]) -> None:
        if data.get("version") != 1:
            raise TaxonomyError("\u4e0d\u652f\u6301\u7684\u5206\u7c7b\u914d\u7f6e\u7248\u672c")
        if not isinstance(data.get("revision"), int) or data["revision"] < 0:
            raise TaxonomyError("\u5206\u7c7b\u914d\u7f6e revision \u65e0\u6548")
        groups = data.get("groups")
        if not isinstance(groups, list) or len(groups) > MAX_GROUPS:
            raise TaxonomyError(f"\u4e00\u7ea7\u9886\u57df\u4e0d\u80fd\u8d85\u8fc7 {MAX_GROUPS} \u4e2a")
        ids: set[str] = set()
        names: set[str] = set()
        for group in groups:
            if not isinstance(group, dict):
                raise TaxonomyError("\u4e00\u7ea7\u9886\u57df\u683c\u5f0f\u65e0\u6548")
            group_id = str(group.get("id") or "")
            source = group.get("source")
            if not group_id or group_id in ids or source not in {"builtin", "custom", "historical"}:
                raise TaxonomyError("\u4e00\u7ea7\u9886\u57df ID \u6216\u6765\u6e90\u65e0\u6548")
            ids.add(group_id)
            name = str(group.get("name") or "")
            checked = name if source == "historical" else _normalized_name(name, 12)
            if checked in names:
                raise TaxonomyError(f"\u4e00\u7ea7\u9886\u57df\u91cd\u590d\uff1a{checked}")
            names.add(checked)
            active = group.get("active")
            categories = group.get("categories")
            if not isinstance(active, bool) or not isinstance(categories, list):
                raise TaxonomyError(f"\u4e00\u7ea7\u9886\u57df\u683c\u5f0f\u65e0\u6548\uff1a{checked}")
            if len(categories) > MAX_CATEGORIES:
                raise TaxonomyError(f"\u201c{checked}\u201d\u4e0b\u7684\u4e8c\u7ea7\u9898\u578b\u8fc7\u591a")
            category_names: set[str] = set()
            for category in categories:
                if not isinstance(category, dict):
                    raise TaxonomyError("\u4e8c\u7ea7\u9898\u578b\u683c\u5f0f\u65e0\u6548")
                category_id = str(category.get("id") or "")
                category_source = category.get("source")
                if (
                    not category_id
                    or category_id in ids
                    or category_source not in {"builtin", "custom", "historical"}
                ):
                    raise TaxonomyError("\u4e8c\u7ea7\u9898\u578b ID \u6216\u6765\u6e90\u65e0\u6548")
                ids.add(category_id)
                category_name = str(category.get("name") or "")
                category_checked = (
                    category_name
                    if category_source == "historical"
                    else _normalized_name(category_name, 24)
                )
                if category_checked in category_names:
                    raise TaxonomyError(f"\u4e8c\u7ea7\u9898\u578b\u91cd\u590d\uff1a{category_checked}")
                category_names.add(category_checked)
                if not isinstance(category.get("active"), bool):
                    raise TaxonomyError(f"\u4e8c\u7ea7\u9898\u578b\u72b6\u6001\u65e0\u6548\uff1a{category_checked}")
                if category["active"] and not active:
                    raise TaxonomyError("\u5df2\u505c\u7528\u7684\u4e00\u7ea7\u9886\u57df\u4e0b\u4e0d\u80fd\u6709\u542f\u7528\u7684\u4e8c\u7ea7\u9898\u578b")

    def _write(self, data: dict[str, Any], *, create_backup: bool = True) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if create_backup and self.path.exists():
            backup_tmp = self.backup_path.with_suffix(".tmp")
            backup_tmp.write_bytes(self.path.read_bytes())
            os.replace(backup_tmp, self.backup_path)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        directory_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _reconcile_historical(
        self,
        historical_pairs: Iterable[tuple[str, str]],
    ) -> None:
        if self.degraded_error:
            return
        known = {
            (str(group["name"]), str(category["name"]))
            for group in self._data["groups"]
            for category in group["categories"]
        }
        changed = False
        for raw_group, raw_category in historical_pairs:
            if (
                not raw_group
                or not raw_category
                or (raw_group == UNCATEGORIZED and raw_category == UNCATEGORIZED)
                or (raw_group, raw_category) in known
            ):
                continue
            group = next(
                (
                    item
                    for item in self._data["groups"]
                    if item["name"] == raw_group
                ),
                None,
            )
            if group is None:
                group = {
                    "id": f"historical-group-{uuid.uuid4().hex}",
                    "name": raw_group,
                    "source": "historical",
                    "active": False,
                    "categories": [],
                }
                self._data["groups"].append(group)
            group["categories"].append(
                {
                    "id": f"historical-category-{uuid.uuid4().hex}",
                    "name": raw_category,
                    "source": "historical",
                    "active": False,
                }
            )
            known.add((raw_group, raw_category))
            changed = True
        if changed:
            candidate = json.loads(json.dumps(self._data, ensure_ascii=False))
            candidate["revision"] += 1
            self._validate(candidate)
            self._write(candidate)
            self._data = candidate

    @contextmanager
    def mutation_guard(self) -> Iterator[None]:
        with self._mutex:
            yield

    def active_payload(self) -> list[dict[str, Any]]:
        with self._mutex:
            return [
                {
                    "name": group["name"],
                    "categories": [
                        category["name"]
                        for category in group["categories"]
                        if category["active"]
                    ],
                }
                for group in self._data["groups"]
                if group["active"]
                and any(category["active"] for category in group["categories"])
            ]

    def active_builtin(self) -> dict[str, tuple[str, ...]]:
        with self._mutex:
            return {
                group["name"]: tuple(
                    category["name"]
                    for category in group["categories"]
                    if category["active"] and category["source"] == "builtin"
                )
                for group in self._data["groups"]
                if group["active"] and group["source"] == "builtin"
            }

    def active_category_names(self) -> list[str]:
        return [
            category
            for categories in self.active_builtin().values()
            for category in categories
        ]

    def is_active_pair(self, group_name: str, category_name: str) -> bool:
        with self._mutex:
            return any(
                group["name"] == group_name
                and group["active"]
                and any(
                    category["name"] == category_name and category["active"]
                    for category in group["categories"]
                )
                for group in self._data["groups"]
            )

    def is_known_pair(self, group_name: str, category_name: str) -> bool:
        with self._mutex:
            return any(
                group["name"] == group_name
                and any(category["name"] == category_name for category in group["categories"])
                for group in self._data["groups"]
            )

    def payload(self, usage: dict[tuple[str, str], int]) -> dict[str, Any]:
        with self._mutex:
            groups = json.loads(json.dumps(self._data["groups"], ensure_ascii=False))
            for group in groups:
                group["usage_count"] = sum(
                    usage.get((group["name"], category["name"]), 0)
                    for category in group["categories"]
                )
                for category in group["categories"]:
                    category["usage_count"] = usage.get(
                        (group["name"], category["name"]), 0
                    )
            return {
                "version": self._data["version"],
                "revision": self._data["revision"],
                "groups": groups,
                "degraded_error": self.degraded_error,
            }

    def update(self, payload: dict[str, Any], expected_revision: int) -> dict[str, Any]:
        if self.degraded_error:
            raise TaxonomyError(f"\u5206\u7c7b\u914d\u7f6e\u5904\u4e8e\u53ea\u8bfb\u964d\u7ea7\u72b6\u6001\uff1a{self.degraded_error}")
        with self._mutex:
            self.lock_path.touch(exist_ok=True)
            with self.lock_path.open("r+") as lock_handle:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    if expected_revision != self._data["revision"]:
                        raise TaxonomyConflict("\u5206\u7c7b\u914d\u7f6e\u5df2\u53d8\u66f4\uff0c\u8bf7\u91cd\u65b0\u6253\u5f00\u5206\u7c7b\u7ba1\u7406")
                    candidate = json.loads(json.dumps(payload, ensure_ascii=False))
                    candidate["version"] = 1
                    candidate["revision"] = expected_revision + 1
                    groups = candidate.get("groups")
                    if not isinstance(groups, list):
                        raise TaxonomyError("\u5206\u7c7b\u914d\u7f6e\u7f3a\u5c11 groups")
                    for group in groups:
                        group.pop("usage_count", None)
                        if group.get("id") is None:
                            group["id"] = f"custom-group-{uuid.uuid4().hex}"
                            group["source"] = "custom"
                        if group.get("source") != "historical":
                            group["name"] = _normalized_name(group.get("name"), 12)
                        for category in group.get("categories", []):
                            category.pop("usage_count", None)
                            if category.get("id") is None:
                                category["id"] = f"custom-category-{uuid.uuid4().hex}"
                                category["source"] = "custom"
                            if category.get("source") != "historical":
                                category["name"] = _normalized_name(
                                    category.get("name"),
                                    24,
                                )
                    old_items: dict[str, tuple[str, str, str | None]] = {}
                    for group in self._data["groups"]:
                        old_items[group["id"]] = (group["name"], group["source"], None)
                        for category in group["categories"]:
                            old_items[category["id"]] = (
                                category["name"],
                                category["source"],
                                group["id"],
                            )
                    new_items: dict[str, tuple[str, str, str | None]] = {}
                    for group in groups:
                        new_items[group["id"]] = (group["name"], group["source"], None)
                        for category in group.get("categories", []):
                            new_items[category["id"]] = (
                                category["name"],
                                category["source"],
                                group["id"],
                            )
                    for item_id, old_value in old_items.items():
                        if new_items.get(item_id) != old_value:
                            raise TaxonomyError("\u73b0\u6709\u5206\u7c7b\u4e0d\u80fd\u5220\u9664\u3001\u6539\u540d\u6216\u79fb\u52a8\uff0c\u53ea\u80fd\u505c\u7528")
                    self._validate(candidate)
                    self._write(candidate)
                    self._data = candidate
                    return json.loads(json.dumps(candidate, ensure_ascii=False))
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
