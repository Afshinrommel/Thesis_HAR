//
//  SimulationService.swift
//  har2
//
//  Created by BEM on 13/02/2026.
//

import Foundation

class SimulationService {
    
    // این سرویس داده‌ها را به ViewModel تحویل می‌دهد
    weak var delegate: SensorProviderDelegate?
    
    private var timer: Timer?
    
    // متغیرهای داخلی که باید حتماً ریست شوند
    private var currentIndex = 0
    private var collectedRows: [[Double]] = [] // ⚠️ بافر خطرناک (منشا باگ)
    private var rows: [String] = []
    
    // سرعت شبیه‌سازی (هر 0.02 ثانیه یک ردیف)
    private let interval = 0.02
    
    // ==========================================
    // 🛑 تابع جدید: ایست و پاکسازی کامل
    // ==========================================
    func stopSimulation() {
        // ۱. توقف تایمر
        timer?.invalidate()
        timer = nil
        
        // ۲. تخلیه بافرها (حل مشکل قاطی کردن نتایج)
        collectedRows.removeAll()
        rows.removeAll()
        currentIndex = 0
        
        print("🧹 [Simulation] Service Fully Reset.")
    }

    // ==========================================
    // تابع شروع شبیه‌سازی
    // ==========================================
    func startSimulation(fileName: String, fileExtension: String = "csv") -> String {
        
        // ۱. اول از همه، همه چیز را پاک کن!
        self.stopSimulation()
        
        // ۲. پیدا کردن مسیر فایل
        guard let path = Bundle.main.path(forResource: fileName, ofType: fileExtension) else {
            return "❌ File \(fileName).\(fileExtension) not found in Bundle!"
        }
        
        do {
            // ۳. خواندن فایل
            let content = try String(contentsOfFile: path, encoding: .utf8)
            self.rows = content.components(separatedBy: "\n")
            
            // تمیزکاری خطوط خالی
            self.rows = self.rows.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            
            // حذف هدر (اگر خط اول متنی است)
            if !self.rows.isEmpty {
                // چک ساده: اگر خط اول حاوی حروف الفبا بود، حذفش کن
                let firstLine = self.rows.first ?? ""
                if firstLine.rangeOfCharacter(from: .letters) != nil {
                    self.rows.removeFirst()
                }
            }
            
            print("📂 [Console] File Loaded: \(self.rows.count) rows.")
            
            // شروع تایمر
            self.startTimer()
            
            return "✅ Simulation Started: \(fileName)"
            
        } catch {
            return "❌ Error loading file: \(error.localizedDescription)"
        }
    }
    
    private func startTimer() {
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            self?.processNextRow()
        }
    }
    
    private func processNextRow() {
        guard currentIndex < rows.count else {
            timer?.invalidate()
            timer = nil
            
            // ✅ صبر می‌کنیم تا آخرین پنجره‌های در حال پردازش، وارد لیست گزارش شوند
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                NotificationCenter.default.post(name: NSNotification.Name("SimulationFinished"), object: nil)
            }
            return
        }
        
        let rowString = rows[currentIndex]
        let columns = rowString.components(separatedBy: ",")
        
        // تبدیل به عدد
        let rowData = columns.compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
        
        // چک کردن تعداد ستون‌ها (حداقل ۱۳ تا لازم داریم)
        if rowData.count >= 13 {
            collectedRows.append(rowData)
            
            // وقتی به ۱۲۸ رسید، ارسال کن
            if collectedRows.count == 128 {
                sendWindowToDelegate()
                
                // Sliding Window: حذف ۶۴ تای اول (Overlap 50%)
                collectedRows.removeFirst(64)
            }
        }
        
        currentIndex += 1
    }
    
    private func sendWindowToDelegate() {
        // استخراج ستون‌ها از بافر جمع شده
        var rolls=[Double](), pitches=[Double](), yaws=[Double]()
        var gravX=[Double](), gravY=[Double](), gravZ=[Double]()
        var userX=[Double](), userY=[Double](), userZ=[Double]() // ستون ۷ تا ۹
        var gyroX=[Double](), gyroY=[Double](), gyroZ=[Double]() // ستون ۱۰ تا ۱۲
        var rawX=[Double](), rawY=[Double](), rawZ=[Double]()
        
        for row in collectedRows {
            // نگاشت ستون‌ها (طبق فایل اکسل شما)
            rolls.append(row[1])
            pitches.append(row[2])
            yaws.append(row[3])
            
            gravX.append(row[4])
            gravY.append(row[5])
            gravZ.append(row[6])
            
            // ⚠️ ترتیب حیاتی مطابق اکسل شما:
            userX.append(row[7])
            userY.append(row[8])
            userZ.append(row[9])
            
            gyroX.append(row[10])
            gyroY.append(row[11])
            gyroZ.append(row[12])
            
            // محاسبه Total Acc برای کدهای قدیمی (اختیاری)
            rawX.append(row[4] + row[7])
            rawY.append(row[5] + row[8])
            rawZ.append(row[6] + row[9])
        }
        
        // ارسال به ViewModel
        DispatchQueue.main.async {
            self.delegate?.didCollectFullWindow(
                rolls: rolls, pitches: pitches, yaws: yaws,
                userAccX: userX, userAccY: userY, userAccZ: userZ,
                gravX: gravX, gravY: gravY, gravZ: gravZ,
                gyroX: gyroX, gyroY: gyroY, gyroZ: gyroZ,
                rawAccX: rawX, rawAccY: rawY, rawAccZ: rawZ
            )
        }
    }
}
