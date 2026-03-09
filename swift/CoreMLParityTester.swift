//
//  CoreMLParityTester.swift
//  har2
//
//  Created by BEM on 17/02/2026.
//

import CoreML
import Foundation

// ۱. ساختار داده برای خواندن اطلاعات از فایل JSON
struct MotionSample: Codable {
    let sample_id: Int
    let real_activity: String
    let top_prediction: String
    let confidence: Double
    let sensor_data: [[Double]]
    let expected_probs: [String: Double]
}

class CoreMLParityTester {
    
    // تابع اصلی برای اجرای تست تطابق (Parity Test)
    func runParityTest() {
        print("🚀 Starting CoreML Parity Test...")
        
        // پیدا کردن فایل JSON در سیستم فایل اپلیکیشن
        guard let url = Bundle.main.url(forResource: "Motion_test_data_tune (1)", withExtension: "json") else {
            print("❌ Error: JSON file not found in bundle. Make sure it's added to the Target Membership.")
            return
        }
        
        do {
            // ۲. دیکود کردن داده‌های JSON
            let data = try Data(contentsOf: url)
            let samples = try JSONDecoder().decode([MotionSample].self, from: data)
            print("✅ Successfully loaded \(samples.count) samples from JSON.")
            
            // ۳. بارگذاری مدل CoreML
            // ⚠️ نکته: اگر نام فایل mlpackage شما چیز دیگری است، نام کلاس زیر را تغییر دهید
            let config = MLModelConfiguration()
            let model = try HAR_Hybrid_NNF_Float16(configuration: config)
            
            var successCount = 0
            
            // ۴. حلقه روی تمام ۲۰ نمونه و انجام پیش‌بینی روی تراشه اپل
            for sample in samples {
                
                // ساخت ماتریس ورودی با ابعاد [1, 128, 9] دقیقاً مثل پایتون
                guard let multiArray = try? MLMultiArray(shape: [1, 128, 9], dataType: .float32) else {
                    fatalError("Failed to create MLMultiArray")
                }
                
                // پر کردن ماتریس با داده‌های سنسور
                for step in 0..<128 {
                    for channel in 0..<9 {
                        let index = [0, NSNumber(value: step), NSNumber(value: channel)]
                        let value = sample.sensor_data[step][channel]
                        multiArray[index] = NSNumber(value: value)
                    }
                }
                
                // انجام پیش‌بینی (نام input_1 باید با ورودی مدل شما در Xcode همخوان باشد)
 
                let input = HAR_Hybrid_NNF_Float16Input(input_1: multiArray) // ✅ درست
                let output = try model.prediction(input: input)
                
                // استخراج نام کلاس پیش‌بینی شده (بسته به تنظیمات، نام متغیر خروجی در Xcode ممکن است classLabel یا target باشد)
                // اگر روی فایل mlpackage در Xcode کلیک کنید، نام دقیق خروجی را می‌بینید
                let predictedLabel = output.classLabel 
                
                // مقایسه نتیجه CoreML با نتیجه Keras
                let isMatch = (predictedLabel == sample.top_prediction)
                if isMatch {
                    successCount += 1
                    print("✅ Sample \(sample.sample_id): Match! (True: \(sample.real_activity), Pred: \(predictedLabel))")
                } else {
                    print("⚠️ Sample \(sample.sample_id): Mismatch! (Keras: \(sample.top_prediction), CoreML: \(predictedLabel))")
                }
            }
            
            // ۵. گزارش نهایی
            print("\n📊 ======================================")
            print("   Parity Test Results: \(successCount)/\(samples.count) matches.")
            if successCount == samples.count {
                print("🏆 100% PARITY ACHIEVED! Model converted perfectly!")
            } else {
                print("⚠️ Some deviations detected. This is normal due to 16-bit compression, but check accuracy.")
            }
            print("======================================\n")
            
        } catch {
            print("❌ Error during parity test: \(error)")
        }
    }
}
