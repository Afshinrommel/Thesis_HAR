//
//  ModelVerifier.swift
//  har2
//
//  Created by BEM on 07/02/2026.
//

import Foundation
import CoreML

// ساختار فایل جیسون
struct TestSample: Codable {
    let source_activity: String
    let python_prediction: String
    let features: [Double] // این همان بردار ۱۲۰ تایی است که پایتون حساب کرده
}

class ModelVerifier {
    
    func runTest() {
        print("\n--- 🧪 STARTING JSON VERIFICATION ---")
        
        // 1. پیدا کردن فایل JSON
        guard let url = Bundle.main.url(forResource: "ios_test_data", withExtension: "json") else {
            print("❌ Error: JSON file not found in bundle!")
            return
        }
        
        do {
            // 2. لود کردن داده‌ها
            let data = try Data(contentsOf: url)
            let samples = try JSONDecoder().decode([TestSample].self, from: data)
              // 3. لود کردن مدل (نام کلاس باید دقیقاً نام فایل مدل شما باشد)
            let rfModel = try MotionSense_120(configuration: MLModelConfiguration())
            
            var passed = 0
            
            for (index, sample) in samples.enumerated() {
                // تبدیل آرایه معمولی به فرمت MLMultiArray
                let mlArray = try MLMultiArray(shape: [120], dataType: .double)
                for (i, val) in sample.features.enumerated() {
                    mlArray[i] = NSNumber(value: val)
                }
                
                // 4. پیش‌بینی
                let input = MotionSense_120Input(features: mlArray) // نام input_1 را در پایتون گذاشتیم
                let output = try rfModel.prediction(input: input)
                
                let iosPrediction = output.classLabel
                
                // 5. مقایسه
                if iosPrediction == sample.python_prediction {
                    print("✅ Sample \(index+1): MATCH (Py: \(sample.python_prediction) | iOS: \(iosPrediction))")
                    passed += 1
                } else {
                    print("❌ Sample \(index+1): MISMATCH! (Py: \(sample.python_prediction) | iOS: \(iosPrediction))")
                }
            }
            
            print("--- RESULT: \(passed)/\(samples.count) passed ---")
            
        } catch {
            print("❌ Critical Error: \(error)")
        }
    }
    
    
    
    func runTestrf() {
            print("\n--- 🧪 STARTING JSON VERIFICATION FOR UCI_RF_90_Vectorized ---")
            
            // ۱. پیدا کردن فایل JSON در پروژه
            guard let url = Bundle.main.url(forResource: "swift_audit_manifest_random", withExtension: "json") else {
                print("❌ Error: JSON file (ios_test_data.json) not found in the project!")
                return
            }
            
            do {
                // ۲. خواندن داده‌های فایل
                let data = try Data(contentsOf: url)
                let samples = try JSONDecoder().decode([TestSample].self, from: data)
                
                // ۳. لود کردن مدل جدید
                let rfModel = try UCI_RF_90_Vectorized(configuration: MLModelConfiguration())
                var passed = 0
                
                for (index, sample) in samples.enumerated() {
                    // تبدیل آرایه ۹۰ تایی جیسون به فرمت قابل فهم برای CoreML
                    let mlArray = try MLMultiArray(shape: [90], dataType: .double)
                    for (i, val) in sample.features.enumerated() {
                        if i < 90 { // اطمینان از اینکه فقط 90 ویژگی اول خوانده شود
                            mlArray[i] = NSNumber(value: val)
                        }
                    }
                    
                    // ۴. تزریق داده‌ها به مدل و دریافت پیش‌بینی
                    let input = UCI_RF_90_VectorizedInput(features_90: mlArray)
                    let output = try rfModel.prediction(input: input)
                    
                    let iosPrediction = output.classLabel
                    
                    // ۵. مقایسه نتیجه iOS با پایتون
                    if iosPrediction == sample.python_prediction {
                        print("✅ Sample \(index+1): MATCH (Py: \(sample.python_prediction) | iOS: \(iosPrediction))")
                        passed += 1
                    } else {
                        print("❌ Sample \(index+1): MISMATCH! (Py: \(sample.python_prediction) | iOS: \(iosPrediction))")
                    }
                }
                
                print("--- RESULT: \(passed)/\(samples.count) PASSED ---\n")
                
            } catch {
                print("❌ Critical Error during verification: \(error)")
            }
        }
}
