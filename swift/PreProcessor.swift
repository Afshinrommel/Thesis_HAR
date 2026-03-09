//
//  PreProcessor.swift
//  har2
//
//  Created by BEM on 05/02/2026.
//

import Foundation

class PreProcessor {
    static let shared = PreProcessor()
    
    // مقادیر اسکیلر V12 Augmented (دقیقاً همان که فرستادید)
    //   private let scalerMin =  [-0.12684103649191247, -0.0554848205345263, -0.03875873267946228, 0.024129953433523088, 0.24791437358106472, -0.027176354969232763, -0.2471210936689341, -0.10814585574693603, -0.009708568499765202]
    
    
    //   private let scalerScale = [0.153882809765537, 0.14822574122496276, 0.12783021759024518, 0.05561569313903055, 0.07184577789982545, 0.08981732140324733, 0.15590156414678738, 0.13657288775518217, 0.12821547947823908]
    
    private let scalerMin = [-0.12689262982862404, -0.0556036323578033, -0.02018312836752023, 0.024256519549471056, 0.24781719316269402, 0.03037830537042341, -0.24523063494361208, -0.1080948569542245, -0.0019711438865578357]
    private let scalerScale = [0.15388033735997422, 0.14824242642045676, 0.12555055383554303, 0.055619720142757303, 0.0718520730272554, 0.08478498529889374, 0.15630164105701747, 0.13656660243420526, 0.1272329607221755]


    
    

    /// مرحله ۱: تغییر ورودی‌ها برای دریافت مستقیم داده‌های تفکیک شده آیفون
    func cleanWindow(
        userAccX: [Double], userAccY: [Double], userAccZ: [Double],
        gyroX: [Double], gyroY: [Double], gyroZ: [Double],
        totalAccX: [Double], totalAccY: [Double], totalAccZ: [Double]
    ) -> [[Double]] {
        
        var cleanedWindow = [[Double]]()
        
        // 🟢 تغییر حیاتی: محاسبه امن حد مجاز برای جلوگیری از کرش
        // این خط تضمین می‌کند که t هیچگاه از تعداد اعضای هیچ‌کدام از آرایه‌ها فراتر نرود
        let limit = [
            userAccX.count, userAccY.count, userAccZ.count,
            gyroX.count, gyroY.count, gyroZ.count,
            totalAccX.count, totalAccY.count, totalAccZ.count
        ].min() ?? 0
        
        // اگر داده‌ها کافی نیستند (کمتر از ۱۲۸)، عملیات را متوقف کن
        if limit < 128 {
            print("⚠️ Warning: Window size is only \(limit). Expected 128.")
            return []
        }

        for t in 0..<limit {
            let rawFrame = [
                userAccY[t], userAccX[t], userAccZ[t], //
                        gyroY[t], gyroX[t], gyroZ[t],          // علامت منفی برا
                        totalAccY[t], totalAccX[t], totalAccZ[t] // علامت منفی برای 
 
            ]
            
            var scaledFrame = [Double]()
            for c in 0..<9 {
                let value = (rawFrame[c] * scalerScale[c]) + scalerMin[c]
                scaledFrame.append(value)
            }
            cleanedWindow.append(scaledFrame)
        }
        return cleanedWindow
    }
    
 }
