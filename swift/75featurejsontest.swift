import Foundation
import CoreML

// =========================================================================
// MODULE: CoreML iOS Verification (Sanity Check) - UPDATED
// DESCRIPTION: Uses dynamic dictionary mapping to bypass generated class limits
// =========================================================================

struct VerificationTestCase: Codable {
    let sample_id: String
    let true_label: String
    let colab_prediction: String
    let features: [Double]
}

class ModelVerifierv{
    
    func runVerification() {
        print("--- Starting iOS Model Verification ---")
        
        guard let url = Bundle.main.url(forResource: "ios_verification_7classes", withExtension: "json") else {
            print("Error: JSON file not found in the App Bundle.")
            return
        }
        
        do {
            // 1. Read and Decode JSON
            let data = try Data(contentsOf: url)
            let testCases = try JSONDecoder().decode([VerificationTestCase].self, from: data)
            print("Successfully loaded \(testCases.count) test cases.\n")
            
            // 2. Initialize the CoreML Model
            let config = MLModelConfiguration()
            let classifier = try RFActivityClassifier(configuration: config)
            
            // 3. Run Predictions using dynamic dictionary
            for test in testCases {
                
                var featureDictionary: [String: Any] = [:]
                
                // Map the 75 features exactly as they were named in Python conversion
                for (index, value) in test.features.enumerated() {
                    let featureName = "feature_\(index)"
                    featureDictionary[featureName] = value
                }
                
                // Create the provider dynamically
                let provider = try MLDictionaryFeatureProvider(dictionary: featureDictionary)
                
                // Perform prediction directly on the underlying MLModel
                let predictionOutput = try classifier.model.prediction(from: provider)
                
                // Extract the predicted class label
                // "classLabel" matches the output_feature_names we set in Colab
                let rawFeature = predictionOutput.featureValue(for: "classLabel")
                let iosPrediction: String

                if rawFeature?.type == .string {
                    iosPrediction = rawFeature?.stringValue ?? "Unknown Error"
                } else if rawFeature?.type == .int64 {
                    iosPrediction = String(rawFeature?.int64Value ?? -1)
                } else {
                    iosPrediction = rawFeature?.description ?? "Type Mismatch"
                }
                
                // Print comparison
                print("ID: \(test.sample_id)")
                print("  - True Label   : \(test.true_label)")
                print("  - Colab Output : \(test.colab_prediction)")
                print("  - iOS Output   : \(iosPrediction)")
                print("--------------------------------------------------")
            }
            
            print("--- Verification Complete ---")
            
        } catch {
            print("Verification failed with error: \(error)")
        }
    }
}
