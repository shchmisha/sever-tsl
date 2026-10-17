#!/usr/bin/env swift
// Local OCR using Apple's installed Vision framework. No API, package or model download.
// Examples:
//   swift tools/vision_ocr.swift --list-languages
//   swift tools/vision_ocr.swift --languages ru-RU,en-US image-01.png image-02.png > ocr.json
// Bounding boxes describe the image AFTER its EXIF orientation has been applied.
// Text order is a geometric heuristic, not a table-layout reconstruction.
import Foundation
import Vision
import ImageIO
import CoreGraphics
import Darwin

struct Rect: Codable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double
    init(_ value: CGRect) {
        x = Double(value.origin.x); y = Double(value.origin.y)
        width = Double(value.width); height = Double(value.height)
    }
}
struct Candidate: Codable {
    let text: String
    let confidence: Float
}
struct TextRegion: Codable {
    let visionIndex: Int
    let text: String
    let confidence: Float
    let boundingBoxNormalizedBottomLeft: Rect
    let boundingBoxPixelsTopLeft: Rect
    let candidates: [Candidate]
}
struct ImageResult: Codable {
    let path: String
    let rawPixelWidth: Int?
    let rawPixelHeight: Int?
    let orientedPixelWidth: Int?
    let orientedPixelHeight: Int?
    let exifOrientation: UInt32?
    let durationSeconds: Double
    let text: String
    let regions: [TextRegion]
    let error: String?
}
struct Configuration: Codable {
    let recognitionLevel: String
    let revision: Int
    let languages: [String]
    let languageCorrection: Bool
    let automaticallyDetectsLanguage: Bool
    let minimumTextHeight: Float
    let candidateCount: Int
}
struct BatchResult: Codable {
    let schemaVersion: Int
    let engine: String
    let operatingSystem: String
    let configuration: Configuration
    let supportedLanguages: [String]
    let images: [ImageResult]
}
struct Options {
    var languages = ["ru-RU", "en-US"]
    var correction = true
    var automaticLanguage = false
    var minimumHeight: Float = 0.005
    var candidates = 3
    var listLanguages = false
    var paths: [String] = []
}
enum UsageError: Error, CustomStringConvertible {
    case message(String)
    var description: String { if case .message(let value) = self { return value }; return "Invalid arguments" }
}
let usage = """
Usage: swift vision_ocr.swift [options] IMAGE [IMAGE ...] > ocr.json
       swift vision_ocr.swift --list-languages

Options:
  --languages ru-RU,en-US  Recognition languages in priority order (default shown)
  --no-correction         Disable language correction; useful for IDs and numbers
  --auto-language         Let Vision detect a language within the configured set
  --min-text-height N     Minimum text height / image height (default 0.005, 0..1)
  --candidates N          Candidates per region (default 3, 1..10)
  --list-languages        Print installed accurate-recognition language identifiers
  --help                  Show this help
  --                      End options; use before a filename starting with '-'

JSON uses both normalized lower-left and pixel upper-left boxes, relative to the
EXIF-oriented image. Combined text uses approximate top-to-bottom, left-to-right
ordering; it does not preserve tables. Exit 2 means some images failed; their
errors remain in the JSON. Recognition is local through macOS Vision.
"""
func parseArguments() throws -> Options {
    var options = Options()
    let args = Array(CommandLine.arguments.dropFirst())
    var index = 0
    var endOptions = false
    func value(_ option: String) throws -> String {
        guard index + 1 < args.count else { throw UsageError.message("Missing value for \(option)") }
        index += 1
        return args[index]
    }
    while index < args.count {
        let arg = args[index]
        if endOptions { options.paths.append(arg) }
        else {
            switch arg {
            case "--": endOptions = true
            case "--help", "-h": print(usage); exit(0)
            case "--list-languages": options.listLanguages = true
            case "--no-correction": options.correction = false
            case "--auto-language": options.automaticLanguage = true
            case "--languages":
                options.languages = try value(arg).split(separator: ",").map(String.init)
                guard !options.languages.isEmpty else { throw UsageError.message("At least one language is required") }
            case "--min-text-height":
                let token = try value(arg)
                guard let number = Float(token), number.isFinite, (0...1).contains(number) else {
                    throw UsageError.message("--min-text-height must be a number between 0 and 1")
                }
                options.minimumHeight = number
            case "--candidates":
                let token = try value(arg)
                guard let number = Int(token), (1...10).contains(number) else {
                    throw UsageError.message("--candidates must be an integer between 1 and 10")
                }
                options.candidates = number
            default:
                guard !arg.hasPrefix("-") else { throw UsageError.message("Unknown option: \(arg)") }
                options.paths.append(arg)
            }
        }
        index += 1
    }
    guard options.listLanguages || !options.paths.isEmpty else { throw UsageError.message("No images supplied\n\(usage)") }
    return options
}
func makeRequest(_ options: Options) -> VNRecognizeTextRequest {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = options.languages
    request.usesLanguageCorrection = options.correction
    request.automaticallyDetectsLanguage = options.automaticLanguage
    request.minimumTextHeight = options.minimumHeight
    return request
}
func recognize(_ path: String, options: Options) -> ImageResult {
    let started = Date()
    let url = URL(fileURLWithPath: path).standardizedFileURL
    var rawWidth: Int? = nil, rawHeight: Int? = nil
    var width: Int? = nil, height: Int? = nil
    var orientationValue: UInt32? = nil
    do {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw UsageError.message("Unable to read image")
        }
        rawWidth = image.width; rawHeight = image.height
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let rawOrientation = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value ?? 1
        let orientation = CGImagePropertyOrientation(rawValue: rawOrientation) ?? .up
        orientationValue = orientation.rawValue
        let swapDimensions = [CGImagePropertyOrientation.leftMirrored, .right, .rightMirrored, .left].contains(orientation)
        let orientedWidth = swapDimensions ? image.height : image.width
        let orientedHeight = swapDimensions ? image.width : image.height
        width = orientedWidth; height = orientedHeight
        let request = makeRequest(options)
        let handler = VNImageRequestHandler(cgImage: image, orientation: orientation, options: [:])
        try handler.perform([request])
        var regions = (request.results ?? []).enumerated().compactMap { (index, observation) -> TextRegion? in
            let candidates = observation.topCandidates(options.candidates).map { Candidate(text: $0.string, confidence: $0.confidence) }
            guard let best = candidates.first else { return nil }
            let box = observation.boundingBox
            let pixels = CGRect(x: box.minX * Double(orientedWidth),
                                y: (1 - box.maxY) * Double(orientedHeight),
                                width: box.width * Double(orientedWidth),
                                height: box.height * Double(orientedHeight))
            return TextRegion(visionIndex: index, text: best.text, confidence: best.confidence,
                              boundingBoxNormalizedBottomLeft: Rect(box), boundingBoxPixelsTopLeft: Rect(pixels),
                              candidates: candidates)
        }
        // Sort by top edge, with source index as a stable tie-breaker.
        regions.sort {
            let a = $0.boundingBoxPixelsTopLeft, b = $1.boundingBoxPixelsTopLeft
            if a.y != b.y { return a.y < b.y }
            if a.x != b.x { return a.x < b.x }
            return $0.visionIndex < $1.visionIndex
        }
        return ImageResult(path: url.path, rawPixelWidth: rawWidth, rawPixelHeight: rawHeight,
                           orientedPixelWidth: width, orientedPixelHeight: height, exifOrientation: orientationValue,
                           durationSeconds: Date().timeIntervalSince(started),
                           text: regions.map(\.text).joined(separator: "\n"), regions: regions, error: nil)
    } catch {
        return ImageResult(path: url.path, rawPixelWidth: rawWidth, rawPixelHeight: rawHeight,
                           orientedPixelWidth: width, orientedPixelHeight: height, exifOrientation: orientationValue,
                           durationSeconds: Date().timeIntervalSince(started), text: "", regions: [],
                           error: String(describing: error))
    }
}
func main() throws {
    let options = try parseArguments()
    let probe = makeRequest(options)
    let supported = try probe.supportedRecognitionLanguages()
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    if options.listLanguages {
        FileHandle.standardOutput.write(try encoder.encode(supported))
        FileHandle.standardOutput.write(Data("\n".utf8))
        return
    }
    let unsupported = options.languages.filter { !supported.contains($0) }
    guard unsupported.isEmpty else {
        throw UsageError.message("Unsupported languages: \(unsupported.joined(separator: ", ")). Use --list-languages.")
    }
    let results = options.paths.map { path in autoreleasepool { recognize(path, options: options) } }
    let configuration = Configuration(recognitionLevel: "accurate", revision: probe.revision,
                                      languages: options.languages, languageCorrection: options.correction,
                                      automaticallyDetectsLanguage: options.automaticLanguage,
                                      minimumTextHeight: options.minimumHeight, candidateCount: options.candidates)
    let result = BatchResult(schemaVersion: 1, engine: "Apple Vision VNRecognizeTextRequest",
                             operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
                             configuration: configuration, supportedLanguages: supported, images: results)
    FileHandle.standardOutput.write(try encoder.encode(result))
    FileHandle.standardOutput.write(Data("\n".utf8))
    if results.contains(where: { $0.error != nil }) { exit(2) }
}
do { try main() }
catch {
    FileHandle.standardError.write(Data("vision_ocr: \(error)\n".utf8))
    exit(1)
}
