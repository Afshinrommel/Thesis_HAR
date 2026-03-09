import Foundation
import CoreML

class UCIVerifier {
    let extractor = FeatureExtractorf()
    
    func runAudit() {
        print("--- 🔍 Starting Parity & Inference Test: swift_audit_manifest_random ---")
        
        guard let path = Bundle.main.path(forResource: "swift_audit_manifest_random", ofType: "json"),
              let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
            print("❌ Error: JSON file not found in Bundle.")
            return
        }
        
        do {
            let samples = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] ?? []
            
            for (index, s) in samples.enumerated() {
                guard let raw = s["input_raw_window"] as? [[Double]],
                      let pyFeats = s["input_features"] as? [Double],
                      let expectedOutput = s["expected_output"] as? [String: Any],
                      let metadata = s["metadata"] as? [String: Any] else { continue }
                
                let activityName = metadata["activity_name"] as? String ?? "Unknown"
                let expectedLabel = expectedOutput["label"] as? String ?? ""
                
                // Transpose Matrix: Python (128x9) -> Swift (9x128)
                var swiftInput = Array(repeating: [Double](repeating: 0, count: 128), count: 9)
                for t in 0..<128 {
                    for c in 0..<9 {
                        swiftInput[c][t] = raw[t][c]
                    }
                }
                
                // Extract Features in Swift (Updated to 75 Features)
                let swiftFeats = extractor.extract75FeaturesUCI(window: swiftInput)
                
                // Compare Features Math (Updated to 75)
                var diff = 0.0
                for i in 0..<75 { diff += abs(swiftFeats[i] - pyFeats[i]) }
                
                print("\nSample \(index + 1) [\(activityName)]:")
                
                if diff < 0.5 {
                    print("✅ Math Parity: OK (Diff: \(diff))")
                    // Run CoreML Inference
                    performInference(features: swiftFeats, expected: expectedLabel)
                } else {
                    print("❌ Math Error: Difference too high (\(diff))")
                }
            }
        } catch {
            print("❌ Error: Could not parse JSON data.")
        }
    }
    
    private func performInference(features: [Double], expected: String) {
        do {
            let config = MLModelConfiguration()
            // Updated model class
            let model = try uci_75_string(configuration: config)
            
            // Updated array shape to 75
            let inputMultiArray = try MLMultiArray(shape: [1, 75], dataType: .double)
            for i in 0..<75 {
                inputMultiArray[i] = NSNumber(value: features[i])
            }
            
            // Updated model input configuration
            let input = uci_75_stringInput(features_75: inputMultiArray)
            let prediction = try model.prediction(input: input)
            
            let predictedLabel = prediction.classLabel
            let probabilities = prediction.classProbability
            
            // --- NORMALIZATION LOGIC ---
            let totalVotes = probabilities.values.reduce(0, +)
            
            let winningVotes = probabilities[predictedLabel] ?? 0.0
            let normalizedConfidence = totalVotes > 0 ? (winningVotes / totalVotes) : 0.0
            // ----------------------------
            
            print("🤖 Model Prediction: \(predictedLabel)")
            print("📈 Confidence Score: \(String(format: "%.2f%%", normalizedConfidence * 100))")
            
            if predictedLabel == expected {
                print("🏁 Status: MATCHED WITH PYTHON")
            } else {
                print("⚠️ Status: MISMATCHED (Python expected: \(expected))")
            }
            
        } catch {
            print("❌ CoreML Inference Error: \(error)")
        }
    }
}
