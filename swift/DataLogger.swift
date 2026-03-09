import Foundation
import CoreMotion

class DataLogger: ObservableObject {
    @Published var isRecording = false
    @Published var recordStatus = "Ready"
    
    
    @Published var lastCapturedURL: URL? = nil
    
    private var fileHandle: FileHandle?
    private var fileURL: URL?
    
    // 🔴 ۱. متغیر داخلی برای ذخیره مقدار پیکر در طول کل زمان ضبط
    private var currentGroundTruth = "Unknown"
    
    func startRecording(activityName: String) {
        lastCapturedURL = nil
        
        // 🔴 ۲. ذخیره مقدار پیکر که از ContentView فرستاده شده است
        self.currentGroundTruth = activityName
        
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let timestamp = formatter.string(from: Date())
        let fileName = "MASTER_\(activityName)_\(timestamp).csv"
        
        guard let docDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        fileURL = docDir.appendingPathComponent(fileName)
        
        // 🔴 ۳. اصلاح هدر: جایگزینی حرف "و" با ویرگول و اضافه کردن ستون آخر
        let header = "Timestamp,Roll,Pitch,Yaw,GravX,GravY,GravZ,UserAccX,UserAccY,UserAccZ,GyroX,GyroY,GyroZ,TotalAccX,TotalAccY,TotalAccZ,Ground_Truth\n"
 
        do {
                    // Ensure any previous handle is closed before starting a new one
                    fileHandle?.closeFile()
                    fileHandle = nil
                    
                    try header.write(to: fileURL!, atomically: true, encoding: .utf8)
                    fileHandle = try FileHandle(forWritingTo: fileURL!)
                    // Removed seekToEndOfFile() because this is a brand new file;
                    // the pointer is naturally at the end of the header.
                    
                    isRecording = true
                    recordStatus = "🔴 Recording: \(fileName)"
                } catch {
                    print("Error creating file: \(error)")
                }catch {
            print("Error creating file: \(error)")
        }
    }
    
    func logFrame(motion: CMDeviceMotion) {
        guard isRecording, let fileHandle = fileHandle else { return }
        
        // 🔴 ۴. استفاده از متغیر داخلی کلاس (حل مشکل ارور viewModel)
        let groundTruth = self.currentGroundTruth
        
        let now = Date().timeIntervalSince1970
        let tx = motion.userAcceleration.x + motion.gravity.x
        let ty = motion.userAcceleration.y + motion.gravity.y
        let tz = motion.userAcceleration.z + motion.gravity.z
        
        // 🔴 ۵. اضافه کردن یک %@ در انتها برای ستون شانزدهم (Ground Truth)
        let line = String(format: "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%@\n",
                          now,
                          motion.attitude.roll, motion.attitude.pitch, motion.attitude.yaw,
                          motion.gravity.x, motion.gravity.y, motion.gravity.z,
                          motion.userAcceleration.x, motion.userAcceleration.y, motion.userAcceleration.z,
                          motion.rotationRate.x, motion.rotationRate.y, motion.rotationRate.z,
                          tx, ty, tz,
                          groundTruth) // اضافه شدن متغیر به انتهای فرمت
        
        if let data = line.data(using: .utf8) {
            fileHandle.write(data)
        }
    }
    
    func stopRecording() {
        fileHandle?.closeFile()
        fileHandle = nil
        isRecording = false
        
        if let validURL = fileURL {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self.lastCapturedURL = validURL
                self.recordStatus = "✅ File Saved. Ready to Share."
            }
        } else {
            recordStatus = "❌ Error: No File URL"
        }
    }
}
