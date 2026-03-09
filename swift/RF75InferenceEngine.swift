import Foundation
import CoreML

class RF75InferenceEngine {
    
    private let extractor = FeatureExtractor75()
    private var model: RFActivityClassifier?
    
    // Label Mapping
    private let labelMapping: [String: String] = [
        "0": "WALK", "1": "UPSTAIRS", "2": "DOWNSTAIRS",
        "3": "SIT", "4": "STAND", "5": "JOG", "6": "LAY"
    ]
    
    // Moving Average Memory
    private var probHistory: [[Double]] = []
    private let smoothingFrames = 5
    private let classOrder = ["WALK", "UPSTAIRS", "DOWNSTAIRS", "SIT", "STAND", "JOG", "LAY"]
    
    init() {
        do {
            let config = MLModelConfiguration()
            self.model = try RFActivityClassifier(configuration: config)
        } catch {
            print("Initialization Error: \(error)")
        }
    }
    
    func runInference(
        userAccX: [Double], userAccY: [Double], userAccZ: [Double],
        rawAccX: [Double], rawAccY: [Double], rawAccZ: [Double],
        gyroX: [Double], gyroY: [Double], gyroZ: [Double],
        completion: @escaping (String, Double, [String: Double]) -> Void
    ) {
        
        
        
            print("--- 🚨 DEBUG RF75 INFERENCE 🚨 ---")
            print("UserAcc Count: \(userAccX.count)")
            print("RawAcc Count: \(rawAccX.count)")
            print("Gyro Count: \(gyroX.count)")
        
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self, let modelWrapper = self.model else { return }
            
            guard let featuresMLArray = self.extractor.extractFeatures(
                userAccX: userAccY, userAccY: userAccX, userAccZ: userAccZ, // 🔴 جای X و Y عوض شد
                rawAccX: rawAccY, rawAccY: rawAccX, rawAccZ: rawAccZ,       // 🔴 جای X و Y عوض شد
                gyroX: gyroY, gyroY: gyroX, gyroZ: gyroZ                    // 🔴 جای X و Y عوض شد
            ) else { return }
            
            do {
                var featureDict: [String: Any] = [:]
                for index in 0..<75 {
                    featureDict["feature_\(index)"] = featuresMLArray[index].doubleValue
                }
                
                let provider = try MLDictionaryFeatureProvider(dictionary: featureDict)
                let output = try modelWrapper.model.prediction(from: provider)
                
                // 1. Extract Raw Label
                var rawLabel = "-1"
                if let val = output.featureValue(for: "classLabel") {
                    if val.type == .string { rawLabel = val.stringValue }
                    else if val.type == .int64 { rawLabel = String(val.int64Value) }
                }
                let finalLabel = self.labelMapping[rawLabel] ?? rawLabel
                
                // 2. Extract CoreML Probabilities
                var probsDict: [String: Double] = [:]
                for featureName in output.featureNames {
                    if let dict = output.featureValue(for: featureName)?.dictionaryValue {
                        for (key, val) in dict {
                            var strKey = "-1"
                            if let kStr = key as? String { strKey = kStr }
                            else if let kInt = key as? Int64 { strKey = String(kInt) }
                            
                            let mappedLabel = self.labelMapping[strKey] ?? strKey
                            probsDict[mappedLabel] = val.doubleValue
                        }
                        break
                    }
                }
                
                // 3. Prepare Current Frame Array
                var currentFrameProbs = [Double](repeating: 0.0, count: 7)
                if probsDict.isEmpty {
                    if let idx = self.classOrder.firstIndex(of: finalLabel) {
                        currentFrameProbs[idx] = 1.0
                    }
                } else {
                    for (i, cls) in self.classOrder.enumerated() {
                        currentFrameProbs[i] = probsDict[cls] ?? 0.0
                    }
                }
                
                // 4. Apply Moving Average Filter
                self.probHistory.append(currentFrameProbs)
                if self.probHistory.count > self.smoothingFrames {
                    self.probHistory.removeFirst()
                }
                
                var smoothedProbs = [Double](repeating: 0.0, count: 7)
                for frame in self.probHistory {
                    for i in 0..<7 {
                        smoothedProbs[i] += frame[i]
                    }
                }
                for i in 0..<7 {
                    smoothedProbs[i] /= Double(self.probHistory.count)
                }
                
                // 5. Reconstruct Smoothed Output
                var finalProbsDict: [String: Double] = [:]
                var maxProb = -1.0
                var smoothedLabel = finalLabel
                
                for (i, cls) in self.classOrder.enumerated() {
                    finalProbsDict[cls] = smoothedProbs[i]
                    if smoothedProbs[i] > maxProb {
                        maxProb = smoothedProbs[i]
                        smoothedLabel = cls
                    }
                }
                
                let realConfidence = maxProb * 100.0
                
                DispatchQueue.main.async {
                    completion(smoothedLabel, realConfidence, finalProbsDict)
                }
            } catch {
                print("CoreML Error: \(error)")
            }
        }
    }
}
