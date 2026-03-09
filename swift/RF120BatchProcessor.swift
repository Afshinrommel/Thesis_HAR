import Foundation
import CoreML

struct CSVManagerRF120Url {
    static func loadDataAndRunInference(fileURL: URL) -> [ProcessedSegment]? {
        do {
            let fileContents = try String(contentsOf: fileURL)
            print("Success: File loaded. Parsing data for RF120...")
            
            var parsedData = [[Double]]()
            let rows = fileContents.components(separatedBy: .newlines)
            
            for row in rows {
                guard !row.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
                let columns = row.components(separatedBy: ",")
                let doubleValues = columns.map { Double($0.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0.0 }
                
                if doubleValues.count >= 15 {
                    parsedData.append(doubleValues)
                }
            }
            
            // Noise reduction: trim 3 seconds (150 rows) from start and end
            let trimCount = 150
            var finalDataToProcess = parsedData
            if parsedData.count > (trimCount * 2) {
                finalDataToProcess = Array(parsedData[trimCount..<(parsedData.count - trimCount)])
            }
            
            let verifier = RF120InferenceVerifierUrl()
            return verifier.runBatchInference(fromCSV: finalDataToProcess)
            
        } catch {
            print("Error: Could not read the file contents.")
            return nil
        }
    }
}

class RF120InferenceVerifierUrl {
    // کلاس‌ها دقیقاً بر اساس مدل ۱۲۰ تایی
    private let classOrder = ["wlk", "ups", "dws", "sit", "std", "jog", "lay"]
    private var model: MotionSense_120?

    init() {
        do {
            let config = MLModelConfiguration()
            self.model = try MotionSense_120(configuration: config)
        } catch {
            print("Error loading RF120 model: \(error)")
        }
    }
    
    // ⭐️ مترجم هوشمند برای اطمینان از تطابق دقیق نام‌ها
    private func getStandardLabel(from rawLabel: String) -> String {
        let upper = rawLabel.uppercased()
        if upper.contains("UP") { return "ups" }
        if upper.contains("DOWN") || upper.contains("DWS") { return "dws" }
        if upper.contains("WALK") || upper.contains("WLK") { return "wlk" }
        if upper.contains("SIT") { return "sit" }
        if upper.contains("STAND") || upper.contains("STD") { return "std" }
        if upper.contains("JOG") || upper.contains("RUN") { return "jog" }
        if upper.contains("LAY") { return "lay" }
        return "wlk"
    }

    func runBatchInference(fromCSV data: [[Double]]) -> [ProcessedSegment] {
        var finalReport: [ProcessedSegment] = []
        guard let model = model else { return finalReport }
        
        // ⭐️ اصلاح حیاتی: هماهنگ‌سازی دقیق ستون‌ها با DataLogger.swift
        let rolls    = data.map { $0.count > 1 ? $0[1] : 0.0 }
        let pitches  = data.map { $0.count > 2 ? $0[2] : 0.0 }
        let yaws     = data.map { $0.count > 3 ? $0[3] : 0.0 }
        let gravX    = data.map { $0.count > 4 ? $0[4] : 0.0 }
        let gravY    = data.map { $0.count > 5 ? $0[5] : 0.0 }
        let gravZ    = data.map { $0.count > 6 ? $0[6] : 0.0 }
        let userAccX = data.map { $0.count > 7 ? $0[7] : 0.0 }
        let userAccY = data.map { $0.count > 8 ? $0[8] : 0.0 }
        let userAccZ = data.map { $0.count > 9 ? $0[9] : 0.0 }
        let gyroX    = data.map { $0.count > 10 ? $0[10] : 0.0 }
        let gyroY    = data.map { $0.count > 11 ? $0[11] : 0.0 }
        let gyroZ    = data.map { $0.count > 12 ? $0[12] : 0.0 }
        
        
        
        
        

        let windowSize = 128
        let stepSize = 64
        var allWindowProbabilities = [[Double]]()

        for start in stride(from: 0, to: data.count - windowSize + 1, by: stepSize) {
            let end = start + windowSize
            
            let features = FeatureExtractor.extractFeatures(
                rolls: Array(rolls[start..<end]),
                pitches: Array(pitches[start..<end]),
                yaws: Array(yaws[start..<end]),
                gravX: Array(gravX[start..<end]),
                gravY: Array(gravY[start..<end]),
                gravZ: Array(gravZ[start..<end]),
                rotX: Array(gyroX[start..<end]),
                rotY: Array(gyroY[start..<end]),
                rotZ: Array(gyroZ[start..<end]),
                userAccX: Array(userAccX[start..<end]),
                userAccY: Array(userAccY[start..<end]),
                userAccZ: Array(userAccZ[start..<end])
            )
            
            do {
                let mlInput = try MLMultiArray(shape: [120], dataType: .double)
                for (i, val) in features.enumerated() { mlInput[i] = NSNumber(value: val) }
                
                let output = try model.prediction(input: MotionSense_120Input(features: mlInput))
                
                var currentFrameProbs = [Double](repeating: 0.0, count: 7)
                var totalVotes = 0.0
                
                // ⭐️ استخراج احتمالات و نرمال‌سازی اصولی
                let probDict = output.classProbability
                for (key, val) in probDict {
                    let mappedLabel = self.getStandardLabel(from: key)
                    if let idx = classOrder.firstIndex(of: mappedLabel) {
                        currentFrameProbs[idx] += val
                        totalVotes += val
                    }
                }
                
                // ⭐️ سیستم نجات (Fallback): اگر احتمالات پوچ بود یا جمعشان صفر شد
                if totalVotes == 0 {
                    let predictedStr = output.classLabel
                    let mappedLabel = self.getStandardLabel(from: predictedStr)
                    if let idx = classOrder.firstIndex(of: mappedLabel) {
                        currentFrameProbs[idx] = 1.0
                        totalVotes = 1.0
                    }
                }
                
                if totalVotes > 0 {
                    currentFrameProbs = currentFrameProbs.map { $0 / totalVotes }
                    allWindowProbabilities.append(currentFrameProbs)
                }
                
            } catch {
                print("RF120 Window error: \(error)")
            }
        }

        if !allWindowProbabilities.isEmpty {
            var avgProbs = [Double](repeating: 0.0, count: 7)
            let numWindows = Double(allWindowProbabilities.count)
            
            for windowProb in allWindowProbabilities {
                for i in 0..<7 { avgProbs[i] += windowProb[i] }
            }
            
            let totalActualSeconds = Double(data.count) * 0.02
            
            for i in 0..<7 {
                let classProbability = avgProbs[i] / numWindows
                let estimatedDuration = totalActualSeconds * classProbability
                
                if estimatedDuration > 0.01 {
                    finalReport.append(ProcessedSegment(
                        activity: getReadableLabel(classOrder[i]),
                        startTime: 0.0,
                        duration: estimatedDuration,
                        type: "Batch Analysis"
                    ))
                }
            }
        }
        return finalReport
    }
    
    private func getReadableLabel(_ label: String) -> String {
        switch label.lowercased() {
        case "wlk": return "WALKING"
        case "ups": return "WALKING UPSTAIRS"
        case "dws": return "WALKING DOWNSTAIRS"
        case "sit": return "SITTING"
        case "std": return "STANDING"
        case "jog": return "JOGGING"
        case "lay": return "LAYING"
        default: return label.uppercased()
        }
    }
}
