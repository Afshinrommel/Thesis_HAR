import Foundation

struct AuditSample: Codable {
    var sample_id: Int
    let sensor_data: [[Double]]
    // --- فیلدهای جدید برای ذخیره نتیجه هوش مصنوعی ---
    let predicted_label: String
    let confidence: Double
    // ---------------------------------------------
    let timestamp: Date
}

class DataRecorder {
    static let shared = DataRecorder()
    private var recordedSamples: [AuditSample] = []
    
    // پاکسازی حافظه (برای شروع ضبط جدید)
    func reset() {
        recordedSamples.removeAll()
        print("📁 DataRecorder: Memory Cleared.")
    }

    // تابع جدید: دریافت لیبل و اطمینان علاوه بر داده سنسور
    func addSample(cleanedWindow: [[Double]], prediction: String, confidence: Double) {
        let sample = AuditSample(
            sample_id: recordedSamples.count,
            sensor_data: cleanedWindow,
            predicted_label: prediction,
            confidence: confidence,
            timestamp: Date()
        )
        recordedSamples.append(sample)
    }
    
    func getCurrentCount() -> Int {
        return recordedSamples.count
    }

    func saveToJSON() -> URL? {
        // حذف ۳ ثانیه اول و ۲ ثانیه آخر
        let skipStart = 3
        let skipEnd = 2
        
        guard recordedSamples.count > (skipStart + skipEnd) else {
            print("⚠️ Data too short to save.")
            return nil
        }

        let trimmedSamples = Array(recordedSamples.dropFirst(skipStart).dropLast(skipEnd))
        var finalSamples: [AuditSample] = []
        for (index, sample) in trimmedSamples.enumerated() {
            var s = sample
            s.sample_id = index // بازنشانی IDها از صفر
            finalSamples.append(s)
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        
        do {
            let data = try encoder.encode(finalSamples)
            let fileName = "Audit_Data_\(Int(Date().timeIntervalSince1970)).json"
            let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(fileName)
            try data.write(to: url)
            print("✅ File saved at: \(url)")
            return url
        } catch {
            print("❌ Error saving JSON: \(error)")
            return nil
        }
    }
}
