import Foundation
import CoreML

struct CSVManagerUCI75Url {
    static func loadDataAndRunInference(fileURL: URL) -> [ProcessedSegment]? {
        do {
            let fileContents = try String(contentsOf: fileURL)
            print("Success: File loaded. Parsing data for UCI75...")
            
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
            
            let verifier = UCI75InferenceVerifierUrl()
            return verifier.runBatchInference(fromCSV: finalDataToProcess)
            
        } catch {
            print("Error: Could not read the file contents.")
            return nil
        }
    }
}

class UCI75InferenceVerifierUrl {
    private let activityLabels = ["DOWNSTAIRS", "LAYING", "SITTING", "STANDING", "UPSTAIRS", "WALKING"]
    private var model: uci_75_string?
    private let extractor = FeatureExtractorf()

    init() {
        do {
            let config = MLModelConfiguration()
            self.model = try uci_75_string(configuration: config)
        } catch {
            print("Error loading uci_75_string model: \(error)")
        }
    }

    func runBatchInference(fromCSV data: [[Double]]) -> [ProcessedSegment] {
        var finalReport: [ProcessedSegment] = []
        guard let model = model else { return finalReport }
        

        
        
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
            
            var transposedWindow = Array(repeating: [Double](repeating: 0, count: windowSize), count: 9)
            
            for t in 0..<windowSize {
                let globalIndex = start + t
                
                // ⚠️ تغییر حیاتی: تطابق کامل با چیدمان پایتون
                // Python: TotAcc(XYZ), BodAcc(XYZ), Gyro(XYZ)
                
                transposedWindow[0][t] = totalAccY[globalIndex] // 🔴 حالا Y رفت جای X
                transposedWindow[1][t] = totalAccX[globalIndex] // 🔴 حالا X رفت جای Y
                transposedWindow[2][t] = totalAccZ[globalIndex]

                transposedWindow[3][t] = userAccY[globalIndex]
                transposedWindow[4][t] = userAccX[globalIndex]
                transposedWindow[5][t] = userAccZ[globalIndex]

                transposedWindow[6][t] = gyroY[globalIndex]
                transposedWindow[7][t] = gyroX[globalIndex]
                transposedWindow[8][t] = gyroZ[globalIndex]
            }

            // ۱. فراخوانی تابع استخراج ۷۵ ویژگی جدید
            let features = extractor.extract75FeaturesUCI(window: transposedWindow)
            
            do {
                // ۲. تغییر سایز تنسور ورودی به ۷۵
                let mlInput = try MLMultiArray(shape: [1, 75], dataType: .double)
                for (i, val) in features.enumerated() {
                    mlInput[i] = NSNumber(value: val)
                }
                
                // ۳. استفاده از کلاس ورودی مدل جدید
                let modelInput = uci_75_stringInput(features_75: mlInput)
                let output = try model.prediction(input: modelInput)
                
                let probDictionary = output.classProbability
                var probs = [Double](repeating: 0.0, count: activityLabels.count)
                var totalVotes = 0.0
                
                for i in 0..<activityLabels.count {
                    if let probValue = probDictionary[activityLabels[i]] {
                        probs[i] = probValue
                        totalVotes += probValue
                    }
                }
                
                if totalVotes > 0 {
                    probs = probs.map { $0 / totalVotes }
                }
                
                allWindowProbabilities.append(probs)
                
            } catch {
                print("UCI75 Window error: \(error)")
            }
        }

        if !allWindowProbabilities.isEmpty {
            var avgProbs = [Double](repeating: 0.0, count: activityLabels.count)
            let numWindows = Double(allWindowProbabilities.count)
            
            for windowProb in allWindowProbabilities {
                for i in 0..<activityLabels.count { avgProbs[i] += windowProb[i] }
            }
            
            let totalActualSeconds = Double(data.count) * 0.02
            
            for i in 0..<activityLabels.count {
                let classProbability = avgProbs[i] / numWindows
                let estimatedDuration = totalActualSeconds * classProbability
                
                if estimatedDuration > 0.01 {
                    finalReport.append(ProcessedSegment(
                        activity: activityLabels[i].uppercased(),
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
