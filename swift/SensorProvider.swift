//
//  SensorProvider.swift
//  har2
//
//  Created by BEM on 05/02/2026.
//

import Foundation
import CoreMotion






// پروتکل برای ارسال داده‌های تفکیک شده (تمیز) و خام (برای مدل‌های مختلف)
protocol SensorProviderDelegate: AnyObject {
    func didCollectFullWindow(
        // ✅ اضافه شدن داده‌های Attitude (حیاتی برای مدل RF جدید)
        rolls: [Double], pitches: [Double], yaws: [Double],
        // داده‌های قبلی حفظ شدند
        userAccX: [Double], userAccY: [Double], userAccZ: [Double],
        gravX: [Double], gravY: [Double], gravZ: [Double],
        gyroX: [Double], gyroY: [Double], gyroZ: [Double],
        rawAccX: [Double], rawAccY: [Double], rawAccZ: [Double]
    )
}

class SensorProvider {
    static let shared = SensorProvider()
    private let motionManager = CMMotionManager()
    weak var delegate: SensorProviderDelegate?
    
    // نمونه از کلاس DataLogger
    let dataLogger = DataLogger()
    
    // صف سریال برای جلوگیری از تداخل داده‌ها
    private let sensorQueue = OperationQueue()
    
    // ✅ بافرهای جدید برای Attitude (Roll, Pitch, Yaw)
    private var rolls = [Double](), pitches = [Double](), yaws = [Double]()
    
    // بافرهای داده‌های تمیز (User Acceleration & Gravity)
    private var userAccX = [Double](), userAccY = [Double](), userAccZ = [Double]()
    private var gravX = [Double](), gravY = [Double](), gravZ = [Double]()
    
    // بافرهای داده‌های خام (Raw Acceleration) برای مدل CNN قدیمی
    private var rawAccX = [Double](), rawAccY = [Double](), rawAccZ = [Double]()
    
    // بافرهای ژیروسکوپ
    private var gyroX = [Double](), gyroY = [Double](), gyroZ = [Double]()
    
    private let windowSize = 128
    private let stepSize = 64
    
    // متغیرهای تست فرکانس (حفظ شده)
    private var lastTime: TimeInterval = 0
    private var sampleCount = 0

    init() {
        sensorQueue.maxConcurrentOperationCount = 1
    }

    func start() {
        
        
        // فعال‌سازی سنسور شتاب‌سنج خام برای مقایسه
        if motionManager.isAccelerometerAvailable {
            motionManager.accelerometerUpdateInterval = 0.02
            motionManager.startAccelerometerUpdates()
        }
        
        
        
        
        guard motionManager.isDeviceMotionAvailable else { return }
        
        // درخواست ۵۰ هرتز (هر ۰.۰۲ ثانیه)
        motionManager.deviceMotionUpdateInterval = 0.02
        
        motionManager.startDeviceMotionUpdates(to: sensorQueue) { [weak self] (data, error) in
            guard let self = self, let motion = data else { return }
            
            // ارسال داده به DataLogger (حفظ شده)
            self.dataLogger.logFrame(motion: motion)
            
            // -------------------------------------------
            // منطق تشخیص زنده (Inference)
            // -------------------------------------------
            
            // --- لاجیک تست فرکانس (حفظ شده) ---
            self.sampleCount += 1
            let now = Date().timeIntervalSince1970
            if now - self.lastTime >= 1.0 {
                // print("📊 Actual Sensor Rate: \(self.sampleCount) Hz") // (کامنت کردم شلوغ نشه، اگر خواستید آنکامنت کنید)
                self.sampleCount = 0
                self.lastTime = now
            }
            
            // ✅ ۱. ذخیره داده‌های Attitude (جدید)
            self.rolls.append(motion.attitude.roll)
            self.pitches.append(motion.attitude.pitch)
            self.yaws.append(motion.attitude.yaw)
            
            // ۲. ذخیره داده‌های تمیز (User Acc)
            self.userAccX.append(motion.userAcceleration.x)
            self.userAccY.append(motion.userAcceleration.y)
            self.userAccZ.append(motion.userAcceleration.z)
            
            // ۳. ذخیره داده‌های جاذبه (Gravity)
            self.gravX.append(motion.gravity.x)
            self.gravY.append(motion.gravity.y)
            self.gravZ.append(motion.gravity.z)
            
            // ۴. ذخیره داده‌های خام (Raw Acc = User + Gravity) - (حفظ شده برای CNN)
            self.rawAccX.append(motion.userAcceleration.x + motion.gravity.x)
            self.rawAccY.append(motion.userAcceleration.y + motion.gravity.y)
            self.rawAccZ.append(motion.userAcceleration.z + motion.gravity.z)
            
            
            
            // --- بخش تست مقایسه (دقیقاً همین‌جا کپی کن) ---
            //        if let hwData = self.motionManager.accelerometerData {
            //  let sumX = motion.userAcceleration.x + motion.gravity.x
            //  let hwX = hwData.acceleration.x
            //  let diff = abs(sumX - hwX)
                
                // چاپ در کنسول برای تحلیل
            //      print("🔍 [Test] Python-Style: \(String(format: "%.4f", sumX)) | Hardware: \(String(format: "%.4f", hwX)) | Delta: \(String(format: "%.4f", diff))")
            //    }
            // ------------------------------------------
            
            
            
            
            
            
            
            // ۵. ذخیره داده‌های ژیروسکوپ
            self.gyroX.append(motion.rotationRate.x)
            self.gyroY.append(motion.rotationRate.y)
            self.gyroZ.append(motion.rotationRate.z)
            
            // بررسی پر شدن پنجره
            // ✅ اصلاح شده و ایمن:
            if self.userAccX.count >= self.windowSize {
                
                // ۱. ابتدا یک کپی (Snapshot) از ۱۲۸ داده تهیه می‌کنیم (قبل از حذف)
                let r = Array(self.rolls.prefix(self.windowSize))
                let p = Array(self.pitches.prefix(self.windowSize))
                let y = Array(self.yaws.prefix(self.windowSize))
                
                let uX = Array(self.userAccX.prefix(self.windowSize))
                let uY = Array(self.userAccY.prefix(self.windowSize))
                let uZ = Array(self.userAccZ.prefix(self.windowSize))
                
                let gX = Array(self.gravX.prefix(self.windowSize))
                let gY = Array(self.gravY.prefix(self.windowSize))
                let gZ = Array(self.gravZ.prefix(self.windowSize))
                
                let gyX = Array(self.gyroX.prefix(self.windowSize))
                let gyY = Array(self.gyroY.prefix(self.windowSize))
                let gyZ = Array(self.gyroZ.prefix(self.windowSize))
                
                let raX = Array(self.rawAccX.prefix(self.windowSize))
                let raY = Array(self.rawAccY.prefix(self.windowSize))
                let raZ = Array(self.rawAccZ.prefix(self.windowSize))

                // ۲. حالا بسته‌های کپی شده را به صف اصلی می‌فرستیم
                DispatchQueue.main.async {
                    self.delegate?.didCollectFullWindow(
                        rolls: r, pitches: p, yaws: y,
                        userAccX: uX, userAccY: uY, userAccZ: uZ,
                        gravX: gX, gravY: gY, gravZ: gZ,
                        gyroX: gyX, gyroY: gyY, gyroZ: gyZ,
                        rawAccX: raX, rawAccY: raY, rawAccZ: raZ
                    )
                }
                
                // ۳. حالا با خیال راحت ۶۴ داده قدیمی را پاک کن (برای ایجاد پنجره لغزان)
                self.removeOldData()
            }
        }
    }
    
    private func removeOldData() {
        // حذف stepSize از تمام آرایه‌ها برای ایجاد همپوشانی (Overlap)
        
        // ✅ حذف قدیمی‌های Attitude
        if rolls.count >= stepSize {
            rolls.removeFirst(stepSize)
            pitches.removeFirst(stepSize)
            yaws.removeFirst(stepSize)
        }
        
        if userAccX.count >= stepSize {
            userAccX.removeFirst(stepSize)
            userAccY.removeFirst(stepSize)
            userAccZ.removeFirst(stepSize)
        }
        if gravX.count >= stepSize {
            gravX.removeFirst(stepSize)
            gravY.removeFirst(stepSize)
            gravZ.removeFirst(stepSize)
        }
        if rawAccX.count >= stepSize {
            rawAccX.removeFirst(stepSize)
            rawAccY.removeFirst(stepSize)
            rawAccZ.removeFirst(stepSize)
        }
        if gyroX.count >= stepSize {
            gyroX.removeFirst(stepSize)
            gyroY.removeFirst(stepSize)
            gyroZ.removeFirst(stepSize)
        }
    }

    func stop() {
        motionManager.stopDeviceMotionUpdates()
        
        // پاکسازی آرایه‌ها
        // ✅ پاکسازی Attitude
        rolls.removeAll(); pitches.removeAll(); yaws.removeAll()
        
        userAccX.removeAll(); userAccY.removeAll(); userAccZ.removeAll()
        gravX.removeAll(); gravY.removeAll(); gravZ.removeAll()
        rawAccX.removeAll(); rawAccY.removeAll(); rawAccZ.removeAll()
        gyroX.removeAll(); gyroY.removeAll(); gyroZ.removeAll()
        
        sampleCount = 0
    }
}
