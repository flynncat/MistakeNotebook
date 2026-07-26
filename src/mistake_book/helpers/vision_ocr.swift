import Foundation
import Vision

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write(Data("usage: vision_ocr.swift IMAGE\n".utf8))
    exit(2)
}

let imageURL = URL(fileURLWithPath: CommandLine.arguments[1])
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
request.recognitionLanguages = ["zh-Hans", "en-US"]
request.minimumTextHeight = 0.008

do {
    let handler = VNImageRequestHandler(url: imageURL, options: [:])
    try handler.perform([request])
    let observations = request.results ?? []
    var lines: [[String: Any]] = []
    for observation in observations {
        let candidates = observation.topCandidates(3)
        guard let candidate = candidates.first else { continue }
        let box = observation.boundingBox
        lines.append([
            "text": candidate.string,
            "confidence": Double(candidate.confidence),
            "alternatives": candidates.map {
                [
                    "text": $0.string,
                    "confidence": Double($0.confidence)
                ]
            },
            "box": [
                Double(box.origin.x),
                Double(box.origin.y),
                Double(box.size.width),
                Double(box.size.height)
            ]
        ])
    }
    lines.sort {
        let left = $0["box"] as! [Double]
        let right = $1["box"] as! [Double]
        if abs(left[1] - right[1]) > 0.02 {
            return left[1] > right[1]
        }
        return left[0] < right[0]
    }
    let payload: [String: Any] = ["lines": lines]
    let data = try JSONSerialization.data(withJSONObject: payload, options: [])
    FileHandle.standardOutput.write(data)
} catch {
    let payload = ["error": error.localizedDescription]
    let data = try JSONSerialization.data(withJSONObject: payload, options: [])
    FileHandle.standardOutput.write(data)
    exit(1)
}
