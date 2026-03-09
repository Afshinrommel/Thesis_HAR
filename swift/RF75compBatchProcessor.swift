import Foundation
import CoreML

struct CSVManagerRF75Url {
    static func loadDataAndRunInference(fileURL: URL) -> [ProcessedSegment]? {
        do {
            let fileContents = try String(contentsOf: fileURL)
            print("Success: File loaded. Parsing data for RF75...")
            
            var parsedData = [[Double]]()
            let rows = fileContents.components(separatedBy: .newlines)
            
            for row in rows {
                guard !row.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
                let columns = row.components(separatedBy: ",")
                let doubleValues = columns.map { Double($0.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0.0 }
                
                if doubleValues.count >= 16 {
                    parsedData.append(doubleValues)
                }
            }
            
            let trimCount = 0
            var finalDataToProcess = parsedData
            if parsedData.count > (trimCount * 2) {
                finalDataToProcess = Array(parsedData[trimCount..<(parsedData.count - trimCount)])
            }
            
            let verifier = RF75InferenceVerifierUrl()
            return verifier.runBatchInference(fromCSV: finalDataToProcess)
            
        } catch {
            print("Error: Could not read the file contents.")
            return nil
        }
    }
}

class RF75InferenceVerifierUrl {
    private let extractor = FeatureExtractor75()
    private var model: RFActivityClassifier?
    
    // ⭐️ لیست کلاس‌ها دقیقاً مطابق نام‌های استاندارد رابط کاربری
    private let classOrder = ["Walking", "Walking Upstairs", "Walking Downstairs", "Sitting", "Standing", "Jogging", "Laying"]

    init() {
        do {
            let config = MLModelConfiguration()
            self.model = try RFActivityClassifier(configuration: config)
        } catch {
            print("❌ Error loading RF75 model: \(error)")
        }
    }
    
    // ⭐️ مترجم هوشمند: خروجی مدل هرچه که باشد، به نام استاندارد تبدیل می‌شود
    private func getStandardLabel(from rawLabel: String) -> String {
        let upper = rawLabel.uppercased()
        if upper.contains("UP") { return "Walking Upstairs" }
        if upper.contains("DOWN") { return "Walking Downstairs" }
        if upper.contains("WALK") { return "Walking" }
        if upper.contains("SIT") { return "Sitting" }
        if upper.contains("STAND") { return "Standing" }
        if upper.contains("JOG") { return "Jogging" }
        if upper.contains("LAY") { return "Laying" }
        
        // اگر مدل عدد خروجی می‌دهد
        if upper == "0" { return "Walking" }
        if upper == "1" { return "Walking Upstairs" }
        if upper == "2" { return "Walking Downstairs" }
        if upper == "3" { return "Sitting" }
        if upper == "4" { return "Standing" }
        if upper == "5" { return "Jogging" }
        if upper == "6" { return "Laying" }
        
        return "Unknown"
    }

    func runBatchInference(fromCSV data: [[Double]]) -> [ProcessedSegment] {
        var finalReport: [ProcessedSegment] = []
        guard let modelWrapper = model else { return finalReport }
        

        
        
        
        

        
        
        let userAccX = data.map { $0.count > 7 ? $0[7] : 0.0 }
        let userAccY = data.map { $0.count > 8 ? $0[8] : 0.0 }
        let userAccZ = data.map { $0.count > 9 ? $0[9] : 0.0 }

        let gyroX = data.map { $0.count > 10 ? $0[10] : 0.0 }
        let gyroY = data.map { $0.count > 11 ? $0[11] : 0.0 }
        let gyroZ = data.map { $0.count > 12 ? $0[12] : 0.0 }

        let totalAccX = data.map { $0.count > 13 ? $0[13] : 0.0 }
        let totalAccY = data.map { $0.count > 14 ? $0[14] : 0.0 }
        let totalAccZ = data.map { $0.count > 15 ? $0[15] : 0.0 }
        
        
        
        

        let windowSize = 128
        let stepSize = 64
        var allWindowProbabilities = [[Double]]()

        for start in stride(from: 0, to: data.count - windowSize + 1, by: stepSize) {
            let end = start + windowSize
            
            guard let featuresMLArray = extractor.extractFeatures(
                userAccX: Array(userAccY[start..<end]), // 🔴 Y به عنوان X فرستاده شد
                userAccY: Array(userAccX[start..<end]), // 🔴 X به عنوان Y فرستاده شد
                userAccZ: Array(userAccZ[start..<end]),
                
                rawAccX: Array(totalAccY[start..<end]), // 🔴 Y به عنوان X فرستاده شد
                rawAccY: Array(totalAccX[start..<end]), // 🔴 X به عنوان Y فرستاده شد
                rawAccZ: Array(totalAccZ[start..<end]),
                
                gyroX: Array(gyroY[start..<end]),       // 🔴 Y به عنوان X فرستاده شد
                gyroY: Array(gyroX[start..<end]),       // 🔴 X به عنوان Y فرستاده شد
                gyroZ: Array(gyroZ[start..<end])
            ) else { continue }
            
            do {
                var featureDict: [String: Any] = [:]
                for index in 0..<75 {
                    featureDict["feature_\(index)"] = featuresMLArray[index].doubleValue
                }
                
                let provider = try MLDictionaryFeatureProvider(dictionary: featureDict)
                let output = try modelWrapper.model.prediction(from: provider)
                
                var currentFrameProbs = [Double](repeating: 0.0, count: 7)
                var totalVotes = 0.0
                
                for featureName in output.featureNames {
                    if let dict = output.featureValue(for: featureName)?.dictionaryValue {
                        for (key, val) in dict {
                            // ⭐️ استفاده از مترجم ضدگلوله
                            let mappedLabel = self.getStandardLabel(from: "\(key)")
                            
                            if let idx = classOrder.firstIndex(of: mappedLabel) {
                                let probValue = (val as? NSNumber)?.doubleValue ?? 0.0
                                currentFrameProbs[idx] = probValue
                                totalVotes += probValue
                            }
                        }
                        break
                    }
                }
                
                if totalVotes > 0 {
                    currentFrameProbs = currentFrameProbs.map { $0 / totalVotes }
                }
                
                allWindowProbabilities.append(currentFrameProbs)
                
            } catch {
                print("❌ RF75 Window error: \(error)")
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
                        activity: classOrder[i],
                        startTime: 0.0,
                        duration: estimatedDuration,
                        type: "Batch Analysis"
                    ))
                }
            }
        }
        return finalReport
    }
}
