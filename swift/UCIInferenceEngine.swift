import Foundation
import CoreML

class UCIInferenceEngine {
    private let extractor = FeatureExtractorf()
    private var model: uci_75_string? // نام مدل جدید شما
    
    private let activityLabels = ["DOWNSTAIRS", "LAYING", "SITTING", "STANDING", "UPSTAIRS", "WALKING"]
    
    init() {
        setupModel()
    }
    
    private func setupModel() {
        do {
            let config = MLModelConfiguration()
            self.model = try uci_75_string(configuration: config)
        } catch {
            print("❌ Error loading CoreML model uci_75_string: \(error)")
        }
    }
    
    func runInference(userAccX: [Double], userAccY: [Double], userAccZ: [Double],
                      gyroX: [Double], gyroY: [Double], gyroZ: [Double],
                      rawAccX: [Double], rawAccY: [Double], rawAccZ: [Double],
                      completion: @escaping (String, Double, [Double]) -> Void) {
        
        guard let model = model else { return }
        
        // ترتیب کانال‌ها مطابق پایتون: TotalAcc(XYZ), BodyAcc(XYZ), Gyro(XYZ)
        // Swap X and Y axes to match iOS CoreMotion with Android dataset coordinate system
                let transposedWindow = [
                    rawAccY, rawAccX, rawAccZ,     // Total Acc (X and Y swapped)
                    userAccY, userAccX, userAccZ,  // Body Acc (X and Y swapped)
                    gyroY, gyroX, gyroZ            // Gyro (X and Y swapped)
                ]
        
        let features = extractor.extract75FeaturesUCI(window: transposedWindow)
        
        do {
            // ساخت ورودی ۷۵ تایی
            let inputMultiArray = try MLMultiArray(shape: [1, 75], dataType: .double)
            for i in 0..<75 {
                inputMultiArray[i] = NSNumber(value: features[i])
            }
            
            // فراخوانی مدل (نام پارامتر ورودی در CoreML معمولاً از متغیر اول گرفته می‌شود)
            let prediction = try model.prediction(input: uci_75_stringInput(features_75: inputMultiArray))
            
            let label = prediction.classLabel
            let probabilities = prediction.classProbability
            
            var probArray = activityLabels.map { probabilities[$0] ?? 0.0 }
            let total = probArray.reduce(0, +)
            if total > 0 { probArray = probArray.map { $0 / total } }
            
            let confidence = (probArray.max() ?? 0.0) * 100.0
            
            DispatchQueue.main.async {
                completion(label, confidence, probArray)
            }
        } catch {
            print("❌ Inference Error: \(error)")
        }
    }
}
