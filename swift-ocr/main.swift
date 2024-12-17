import Foundation
import Logging
import ArgumentParser
import AppKit
import Vision

public enum Errors: Error {
    case notFound
    case notDirectory
    case couldNotCreateOutput
    case invalidImage
    case cgImage
    case processError
    case unsupportedOS
}

@available(macOS 10.15, *)
struct SwiftOCR: ParsableCommand {
    
    @Argument(help:"The path to a directory of image files to extract text from")
    var inputDirectory: String
    
    @Argument(help:"The path to the JSONL file to save the extracted text")
    var outputFile: String
    
    func run() throws {
        let logger = Logger(label: "TextExtractCLI")
        
        logger.info("Input directory: \(inputDirectory)")
        logger.info("Output file: \(outputFile)")
        
        let fm = FileManager.default
        
        var isDir: ObjCBool = false
        
        if (!fm.fileExists(atPath: inputDirectory, isDirectory: &isDir)){
            throw(Errors.notFound)
        } else if (!isDir.boolValue) {
            throw(Errors.notDirectory)
        }
        
        let subdirs = try fm.contentsOfDirectory(atPath: inputDirectory)
        
        try "".write(to: URL(fileURLWithPath: outputFile), atomically: true, encoding: .utf8)
        
        if let fileHandle = FileHandle(forWritingAtPath: outputFile) {
            defer {
                fileHandle.closeFile()
            } // Ensure the file is closed afterward
            for subdir in subdirs {
                let subdirPath = NSString.path(withComponents: [inputDirectory, subdir])
                logger.info("Working in \"\(subdirPath)\"")
                
                let files = try fm.contentsOfDirectory(atPath: subdirPath)
                
                let images = files.filter { $0.hasSuffix(".jpg") || $0.hasSuffix(".png") }
                
                for imPath in images {
                    var detectionsDict: [String: [[String: Any]]] = [:]
                    
                    let path = NSString.path(withComponents: [subdirPath, imPath])
                    logger.info("Processing \"\(path)\"")
                    guard let im = NSImage(byReferencingFile:path) else {
                        throw(Errors.invalidImage)
                    }
                    
                    guard let cgImage = im.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                        throw(Errors.cgImage)
                    }
                    
                    let rsp = ProcessImage(image: cgImage)
                    
                    switch rsp {
                    case .failure(let error):
                        throw(error)
                    case .success(let detections):
                        let width = Float(cgImage.width)
                        let height = Float(cgImage.height)
                        // If there are no detections, continue
                        if detections.isEmpty { continue }
                        detectionsDict["detections"] = detections.map({ (text, rect, confidence) in
                            let x = Float(rect.minX)
                            let y = Float(rect.minY)
                            let w = Float(rect.maxX - rect.minX)
                            let h = Float(rect.maxY - rect.minY)
                            return [
                                "image": imPath,
                                "text": text,
                                "x": x,
                                "y": y,
                                "w": w,
                                "h": h,
                                "relX": x / width,
                                "relY": y / height,
                                "relW": w / width,
                                "relH": h / height,
                                "confidence": confidence
                            ]
                        })
                        let jsonData = try JSONSerialization.data(withJSONObject: detectionsDict)
                        let jsonString = String(bytes: jsonData, encoding: .utf8)!
                        fileHandle.write((jsonString + "\n").data(using: .utf8)!)
                    }
                }
            }
        } else {
            print("Error: Could not open file for writing.")
        }
        
        
    }
    
    func ProcessImage(image: CGImage) -> Result<[(String, CGRect, VNConfidence)], Error> {
        let requestHandler = VNImageRequestHandler(cgImage: image)
        
        let request = VNRecognizeTextRequest()
        
        do {
            try requestHandler.perform([request])
        } catch {
            return .failure(error)
        }
        
        var detections = [(String, CGRect, VNConfidence)]()
        
        if request.results != nil {
            
            let maximumCandidates = 1
            for observation in request.results! {
                guard let candidate = observation.topCandidates(maximumCandidates).first else { continue }
                // Find the bounding-box observation for the string range.
                let stringRange = candidate.string.startIndex..<candidate.string.endIndex
                let boxObservation = try? candidate.boundingBox(for: stringRange)
                
                // Get the normalized CGRect value.
                let boundingBox = boxObservation?.boundingBox ?? .zero
                
                // Convert the rectangle from normalized coordinates to image coordinates.
                let boundingRect = VNImageRectForNormalizedRect(boundingBox, Int(image.width), Int(image.height))
                
                detections.append((candidate.string, boundingRect, candidate.confidence))
            }
        }
        
        return .success(detections)
    }
}

if #available(macOS 10.15, *) {
    SwiftOCR.main()
} else {
    throw(Errors.unsupportedOS)
}
