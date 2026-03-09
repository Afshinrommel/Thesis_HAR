import Foundation
import CoreML

class NeuralInferenceVerifierUrl {
    
    let classes = ["Jogging", "Sitting", "Standing", "Walking", "Walking Downstairs", "Walking Upstairs"]
    private var model: HAR_Hybrid_NNF_Float16?

    init() {
        do {
            let config = MLModelConfiguration()
            self.model = try HAR_Hybrid_NNF_Float16(configuration: config)
        } catch {
            print("Error loading CoreML model: \(error)")
        }
    }

    func runBatchInference(fromCSV data: [[Double]]) -> [ProcessedSegment] {
        var finalReport: [ProcessedSegment] = []
        
        guard let model = model else {
            print("Model not initialized.")
            return finalReport
        }
        
        let preProcessor = PreProcessor.shared
        
        // Safety checks added to array indices to prevent crashes
        let userAccX = data.map { $0.count > 7 ? $0[7] : 0.0 }
        let userAccY = data.map { $0.count > 8 ? $0[8] : 0.0 }
        let userAccZ = data.map { $0.count > 9 ? $0[9] : 0.0 }
        let gyroX = data.map { $0.count > 10 ? $0[10] : 0.0 }
        let gyroY = data.map { $0.count > 11 ? $0[11] : 0.0 }
        let gyroZ = data.map { $0.count > 12 ? $0[12] : 0.0 }
        let totalAccX = data.map { ($0.count > 7 && $0.count > 4) ? $0[4] + $0[7] : 0.0 }
        let totalAccY = data.map { ($0.count > 8 && $0.count > 5) ? $0[5] + $0[8] : 0.0 }
        let totalAccZ = data.map { ($0.count > 9 && $0.count > 6) ? $0[6] + $0[9] : 0.0 }

        let scaledFeatures = preProcessor.cleanWindow(
            userAccX: userAccX, userAccY: userAccY, userAccZ: userAccZ,
            gyroX: gyroX, gyroY: gyroY, gyroZ: gyroZ,
            totalAccX: totalAccX, totalAccY: totalAccY, totalAccZ: totalAccZ
        )

        let windowSize = 128
        let stepSize = 64
        var allWindowProbabilities = [[Double]]()

        for start in stride(from: 0, to: scaledFeatures.count - windowSize + 1, by: stepSize) {
            let window = Array(scaledFeatures[start..<start + windowSize])
            
            do {
                let inputShape = [1, NSNumber(value: windowSize), 9] as [NSNumber]
                let inputMultiArray = try MLMultiArray(shape: inputShape, dataType: .double)
                
                for t in 0..<windowSize {
                    for c in 0..<9 {
                        let index = [0, t, c] as [NSNumber]
                        inputMultiArray[index] = NSNumber(value: window[t][c])
                    }
                }
                
                let modelInput = HAR_Hybrid_NNF_Float16Input(input_1: inputMultiArray)
                let output = try model.prediction(input: modelInput)
                
                guard let probDictionary = output.featureValue(for: "Identity")?.dictionaryValue else { continue }
                
                var probs = [Double](repeating: 0.0, count: classes.count)
                for i in 0..<classes.count {
                    let className = classes[i]
                    if let probNumber = probDictionary[className] as? NSNumber {
                        probs[i] = probNumber.doubleValue
                    }
                }
                allWindowProbabilities.append(probs)
                
            } catch {
                print("Error at window \(start): \(error)")
            }
        }

        if !allWindowProbabilities.isEmpty {
            var avgProbs = [Double](repeating: 0.0, count: classes.count)
            let numWindows = Double(allWindowProbabilities.count)
            
            for windowProb in allWindowProbabilities {
                for i in 0..<classes.count { avgProbs[i] += windowProb[i] }
            }
            
            // Calculate the actual total duration of the file based on rows (50Hz = 0.02s)
            let totalActualSeconds = Double(data.count) * 0.02
            
            // Generate clean segments for the UI Report to calculate percentages automatically
            for i in 0..<classes.count {
                let classProbability = avgProbs[i] / numWindows
                
                // Convert probability to estimated time in seconds
                let estimatedDuration = totalActualSeconds * classProbability
                
                // Only include activities that actually occurred to keep the list clean
                if estimatedDuration > 0.01 {
                    finalReport.append(ProcessedSegment(
                        activity: classes[i].uppercased(),
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
