import Foundation
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
struct TextExtractCLI: ParsableCommand {
    
    @Argument(help:"The path to a directory of image files to extract text from")
    var inputDirectory: String
    
    @Argument(help:"The path to a directory where output JSON files should be saved")
    var outputDirectory: String
    
    func run() throws {
        print("Input directory: \(inputDirectory)")
        print("Output directory: \(outputDirectory)")
        
        let fm = FileManager.default
        
        var isDir: ObjCBool = false
        
        if (!fm.fileExists(atPath: inputDirectory, isDirectory: &isDir)){
            throw(Errors.notFound)
        } else if (!isDir.boolValue) {
            throw(Errors.notDirectory)
        }
        
        if (!fm.fileExists(atPath: outputDirectory, isDirectory: &isDir)) {
            print("Creating output directory")
        } else if (isDir.boolValue) {
            print("Output directory already exists")
        } else {
            throw(Errors.couldNotCreateOutput)
        }
        
        let files = try fm.contentsOfDirectory(atPath: inputDirectory)
        
        let images = files.filter { $0.hasSuffix(".jpg") || $0.hasSuffix(".png") }
        
        var detectionsDict: [String: [[String: Any]]] = [:]
        
        for imPath in images {
            let path = NSString.path(withComponents: [inputDirectory, imPath])
            print("Processing \"\(path)\"")
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
                detectionsDict[imPath] = detections.map({ (text, rect, confidence) in
                    [
                        "text": text,
                        "minX": rect.minX,
                        "maxX": rect.maxX,
                        "minY": rect.minY,
                        "maxY": rect.maxY,
                        "confidence": confidence
                    ]
                })
            }
        }
        
        let jsonData = try JSONSerialization.data(withJSONObject: detectionsDict, options: .prettyPrinted)
        print(String(bytes: jsonData, encoding: .utf8)!)
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
    TextExtractCLI.main()
} else {
    throw(Errors.unsupportedOS)
}
