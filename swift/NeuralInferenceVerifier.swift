import Foundation
import CoreML

class NeuralInferenceVerifier {
    
    // 1. List of classes (Exact order from Python training)
    let classes = ["Jogging", "Sitting", "Standing", "Walking", "Walking Downstairs", "Walking Upstairs"]
    
    // 2. Model instance identified in Xcode
    private var model: HAR_Hybrid_NNF_Float16?

    init() {
        do {
            let config = MLModelConfiguration()
            self.model = try HAR_Hybrid_NNF_Float16(configuration: config)
        } catch {
            print("Error loading CoreML model: \(error)")
        }
    }

    func runBatchInference(fromCSV data: [[Double]]) {
        guard let model = model else {
            print("Model not initialized.")
            return
        }
        
        let preProcessor = PreProcessor.shared
        
        // 3. Extract columns based on standard Excel indices
        let userAccX = data.map { $0[7] }
        let userAccY = data.map { $0[8] }
        let userAccZ = data.map { $0[9] }
        let gyroX = data.map { $0[10] }
        let gyroY = data.map { $0[11] }
        let gyroZ = data.map { $0[12] }
        let totalAccX = data.map { $0[4] + $0[7] }
        let totalAccY = data.map { $0[5] + $0[8] }
        let totalAccZ = data.map { $0[6] + $0[9] }

        // 4. Preprocessing (Scaling -1 to 1 and Y-axis inversion)
        let scaledFeatures = preProcessor.cleanWindow(
            userAccX: userAccX, userAccY: userAccY, userAccZ: userAccZ,
            gyroX: gyroX, gyroY: gyroY, gyroZ: gyroZ,
            totalAccX: totalAccX, totalAccY: totalAccY, totalAccZ: totalAccZ
        )

        let windowSize = 128
        let stepSize = 64
        var allWindowProbabilities = [[Double]]()

        print("Running Neural Inference (Parity Mode)...")

        // 5. Window-based inference loop
        for start in stride(from: 0, to: scaledFeatures.count - windowSize + 1, by: stepSize) {
            let window = Array(scaledFeatures[start..<start + windowSize])
            
            do {
                // a. Define input shape [1, 128, 9]
                let inputShape = [1, NSNumber(value: windowSize), 9] as [NSNumber]
                let inputMultiArray = try MLMultiArray(shape: inputShape, dataType: .double)
                
                // b. Fill matrix (Row-Major)
                for t in 0..<windowSize {
                    for c in 0..<9 {
                        let index = [0, t, c] as [NSNumber]
                        inputMultiArray[index] = NSNumber(value: window[t][c])
                    }
                }
                
                // c. Execute model
                let modelInput = HAR_Hybrid_NNF_Float16Input(input_1: inputMultiArray)
                let output = try model.prediction(input: modelInput)
                
                // d. Safely extract dictionary bypassing raw MLMultiArray
                guard let probDictionary = output.featureValue(for: "Identity")?.dictionaryValue else {
                    print("Failed to extract dictionary for Identity at window \(start)")
                    continue
                }
                
                var probs = [Double](repeating: 0.0, count: classes.count)
                
                // e. Extract probabilities using the class names as keys
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

        // 6. Global Averaging (Replicating Colab logic)
        if !allWindowProbabilities.isEmpty {
            var avgProbs = [Double](repeating: 0.0, count: classes.count)
            let numWindows = Double(allWindowProbabilities.count)
            
            for windowProb in allWindowProbabilities {
                for i in 0..<classes.count {
                    avgProbs[i] += windowProb[i]
                }
            }
            
            print("\n--- Results Summary (Match with Colab) ---")
            for i in 0..<classes.count {
                let finalPercentage = (avgProbs[i] / numWindows) * 100
                print(String(format: "   - %-18@: %6.2f%%", classes[i], finalPercentage))
            }
            
            if let winnerIdx = avgProbs.enumerated().max(by: { $0.element < $1.element })?.offset {
                print("\nWINNER: \(classes[winnerIdx])")
            }
        }
    }
}
