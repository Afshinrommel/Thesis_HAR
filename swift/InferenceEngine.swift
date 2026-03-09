//
//  InferenceEngine.swift
//  har2
//
//  Created by BEM on 05/02/2026.
//

import Foundation
import CoreML




struct RecognitionResult {
    let label: String
    let confidence: Double
    let allProbs: [String: Double]
}

class InferenceEngine {
    static let shared = InferenceEngine()
    private var model: HAR_Hybrid_NNF_Float16?
    
  


    init() {
        let config = MLModelConfiguration()
        config.computeUnits = .all // استفاده از تمام قدرت پردازشی (GPU/Neural Engine)
        self.model = try? HAR_Hybrid_NNF_Float16(configuration: config)
    }

    /// اجرای پیش‌بینی روی داده‌های تمیز شده
    func predict(cleanedWindow: [[Double]]) -> RecognitionResult? {
        guard let model = model else { return nil }
        
        do {
            // ۱. تبدیل آرایه دو بعدی به MLMultiArray (1x128x9)
            let inputArray = try MLMultiArray(shape: [1, 128, 9] as [NSNumber], dataType: .float32)
            
            for (t, row) in cleanedWindow.enumerated() {
                for (c, value) in row.enumerated() {
                    let key = [0, t, c] as [NSNumber]
                    inputArray[key] = NSNumber(value: Float(value))
                }
            }
            
            // ۲. اجرای مدل
            let output = try model.prediction(input_1: inputArray)
            
            // ۳. استخراج نتایج
            let label = output.classLabel
            let allProbs = output.Identity
            let confidence = (allProbs[label] ?? 0.0) * 100.0
            
            return RecognitionResult(label: label, confidence: confidence, allProbs: allProbs)
            
        } catch {
            print("❌ Inference Error: \(error)")
            return nil
        }
    }
}
