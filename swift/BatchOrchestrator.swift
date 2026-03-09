import Foundation
import CoreML

struct ModelWinningResult: Identifiable {
    let id = UUID()
    let modelName: String
    let winningClass: String
    let confidence: Double
    var sourceFile: String = "Captured"
}

class BatchOrchestrator {
    static let shared = BatchOrchestrator()
    
    private let cnnClasses = ["Jogging", "Sitting", "Standing", "Walking", "Walking Downstairs", "Walking Upstairs"]
    private let rf120Classes = ["dws", "jog", "sit", "std", "ups", "wlk"]
    private let rf75HybridMap = ["walk", "upstairs", "downstairs", "sit", "stand", "jog", "lay"]

    func runFullComparison(fileURL: URL, fileName: String = "Unknown") -> [ModelWinningResult] {
        var results: [ModelWinningResult] = []
        let parsedData = parseCSV(fileURL: fileURL)
        guard !parsedData.isEmpty else { return results }
        
        if let cnnRes = runCNNParity(data: parsedData) {
            results.append(ModelWinningResult(modelName: "cnn-attention", winningClass: cnnRes.label.lowercased(), confidence: cnnRes.conf, sourceFile: fileName))
        }
        
        if let rf120Res = runRF120Parity(data: parsedData) {
            results.append(ModelWinningResult(modelName: "rf-120 (ms)", winningClass: rf120Res.label.lowercased(), confidence: rf120Res.conf, sourceFile: fileName))
        }
        
        // اصلاح شده: مدل هیبرید حالا درصد واقعی می‌دهد
        if let hybridRes = runRF75HybridParity(data: parsedData) {
            results.append(ModelWinningResult(modelName: "rf-75 (hybrid)", winningClass: hybridRes.label.lowercased(), confidence: hybridRes.conf, sourceFile: fileName))
        }
        
        if let uciRes = runRF75UCIParity(data: parsedData) {
            results.append(ModelWinningResult(modelName: "rf-75 (uci)", winningClass: uciRes.label.lowercased(), confidence: uciRes.conf, sourceFile: fileName))
        }
        
        return results
    }

    // MARK: - اصلاح شده: مدل RF-75 Hybrid برای درصد واقعی
    private func runRF75HybridParity(data: [[Double]]) -> (label: String, conf: Double)? {
            let extractor75 = FeatureExtractor75()
            guard let classifier = try? RFActivityClassifier(configuration: MLModelConfiguration()) else { return nil }
            
            var cumulativeScores = [String: Double]()
            var totalOverallSum = 0.0 // برای نرمال‌سازی نهایی
            
            for start in stride(from: 0, to: data.count - 128 + 1, by: 64) {
                let win = Array(data[start..<start + 128])
                
                // 🟢 اعمال نگاشت (Swap X & Y): جابه‌جایی ایندکس‌های ستون‌ها برای هم‌ترازی با مدل پایتون
                if let mlMA = extractor75.extractFeatures(
                    userAccX: win.map { $0[8] },           // محور Y گوشی به عنوان X مدل (ایندکس 8)
                    userAccY: win.map { $0[7] },           // محور X گوشی به عنوان Y مدل (ایندکس 7)
                    userAccZ: win.map { $0[9] },
                    
                    rawAccX:  win.map { $0[5] + $0[8] },  // مجموع Y گوشی به عنوان X مدل (ایندکس 5 و 8)
                    rawAccY:  win.map { $0[4] + $0[7] },  // مجموع X گوشی به عنوان Y مدل (ایندکس 4 و 7)
                    rawAccZ:  win.map { $0[6] + $0[9] },
                    
                    gyroX:    win.map { $0[11] },          // ژیروسکوپ Y به عنوان X مدل (ایندکس 11)
                    gyroY:    win.map { $0[10] },          // ژیروسکوپ X به عنوان Y مدل (ایندکس 10)
                    gyroZ:    win.map { $0[12] }
                ) {
                    
                    var dict: [String: Any] = [:]
                    for i in 0..<75 { dict["feature_\(i)"] = mlMA[i].doubleValue }
                    
                    if let provider = try? MLDictionaryFeatureProvider(dictionary: dict),
                       let out = try? classifier.model.prediction(from: provider) {
                        
                        // استخراج تمام احتمالات/آراء این پنجره
                        if let windowProbs = out.featureValue(for: "classProbability")?.dictionaryValue as? [String: Double] {
                            for (activity, score) in windowProbs {
                                cumulativeScores[activity, default: 0.0] += score
                                totalOverallSum += score
                            }
                        }
                    }
                }
            }
            
            // پیدا کردن فعالیت با بالاترین مجموع امتیاز
            guard totalOverallSum > 0, let best = cumulativeScores.max(by: { $0.value < $1.value }) else { return nil }
            
            // محاسبه درصد واقعی
            let finalConfidence = (best.value / totalOverallSum) * 100.0
            
            var name = best.key
            // مپینگ عدد به نام فعالیت در صورت نیاز
            if let idx = Int(name), idx >= 0 && idx < rf75HybridMap.count {
                name = rf75HybridMap[idx]
            }
            
            return (label: name, conf: finalConfidence)
        }

    // MARK: - بقیه توابع بدون تغییر برای حفظ ثبات
    private func runCNNParity(data: [[Double]]) -> (label: String, conf: Double)? {
        let uX = data.map { $0[7] }; let uY = data.map { $0[8] }; let uZ = data.map { $0[9] }
        let gX = data.map { $0[10] }; let gY = data.map { $0[11] }; let gZ = data.map { $0[12] }
        let tX = data.map { $0[4] + $0[7] }; let tY = data.map { $0[5] + $0[8] }; let tZ = data.map { $0[6] + $0[9] }
        let scaled = PreProcessor.shared.cleanWindow(userAccX: uX, userAccY: uY, userAccZ: uZ, gyroX: gX, gyroY: gY, gyroZ: gZ, totalAccX: tX, totalAccY: tY, totalAccZ: tZ)
        var sumProbs = [Double](repeating: 0.0, count: cnnClasses.count)
        for start in stride(from: 0, to: scaled.count - 128 + 1, by: 64) {
            let window = Array(scaled[start..<start + 128])
            if let model = try? HAR_Hybrid_NNF_Float16(configuration: MLModelConfiguration()),
               let inputMA = try? MLMultiArray(shape: [1, 128, 9], dataType: .double) {
                for t in 0..<128 { for c in 0..<9 { inputMA[[0, t, c] as [NSNumber]] = NSNumber(value: window[t][c]) } }
                if let out = try? model.prediction(input: HAR_Hybrid_NNF_Float16Input(input_1: inputMA)),
                   let dict = out.featureValue(for: "Identity")?.dictionaryValue {
                    for i in 0..<cnnClasses.count { sumProbs[i] += (dict[cnnClasses[i]] as? NSNumber)?.doubleValue ?? 0.0 }
                }
            }
        }
        return getNormalizedWinner(sumProbs: sumProbs, classes: cnnClasses)
    }

    private func runRF120Parity(data: [[Double]]) -> (label: String, conf: Double)? {
        var sumProbs = [Double](repeating: 0.0, count: rf120Classes.count)
        for start in stride(from: 0, to: data.count - 128 + 1, by: 64) {
            let win = Array(data[start..<start + 128])
            let features = FeatureExtractor.extractFeatures(rolls: win.map{$0[0]}, pitches: win.map{$0[1]}, yaws: win.map{$0[2]}, gravX: win.map{$0[4]}, gravY: win.map{$0[5]}, gravZ: win.map{$0[6]}, rotX: win.map{$0[10]}, rotY: win.map{$0[11]}, rotZ: win.map{$0[12]}, userAccX: win.map{$0[7]}, userAccY: win.map{$0[8]}, userAccZ: win.map{$0[9]})
            if let model = try? MotionSense_120(configuration: MLModelConfiguration()), let mlInput = try? MLMultiArray(shape: [120], dataType: .double) {
                for (i, v) in features.enumerated() { mlInput[i] = NSNumber(value: v) }
                if let out = try? model.prediction(input: MotionSense_120Input(features: mlInput)) {
                    for i in 0..<rf120Classes.count { sumProbs[i] += out.classProbability[rf120Classes[i]] ?? 0.0 }
                }
            }
        }
        return getNormalizedWinner(sumProbs: sumProbs, classes: rf120Classes)
    }

    private func runRF75UCIParity(data: [[Double]]) -> (label: String, conf: Double)? {
        let extractorUCI = FeatureExtractorf()
        guard let model = try? uci_75_string(configuration: MLModelConfiguration()) else { return nil }
        var classSums = [String: Double]()
        var totalVotes = 0.0
        for start in stride(from: 0, to: data.count - 128 + 1, by: 64) {
            let win = Array(data[start..<start + 128])
            let channels = [win.map{$0[5]+$0[8]}, win.map{$0[4]+$0[7]}, win.map{$0[6]+$0[9]}, win.map{$0[8]}, win.map{$0[7]}, win.map{$0[9]}, win.map{$0[11]}, win.map{$0[10]}, win.map{$0[12]}]
            let features = extractorUCI.extract75FeaturesUCI(window: channels)
            do {
                let mlInput = try MLMultiArray(shape: [1, 75], dataType: .double)
                for i in 0..<75 { mlInput[i] = NSNumber(value: features[i]) }
                let out = try model.prediction(input: uci_75_stringInput(features_75: mlInput))
                for (activity, score) in out.classProbability {
                    classSums[activity, default: 0.0] += score
                    totalVotes += score
                }
            } catch { continue }
        }
        guard totalVotes > 0, let best = classSums.max(by: { $0.value < $1.value }) else { return nil }
        return (label: best.key, conf: (best.value / totalVotes) * 100.0)
    }

    private func parseCSV(fileURL: URL) -> [[Double]] {
        do {
            let content = try String(contentsOf: fileURL)
            let rows = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
            return rows.compactMap { row in
                let columns = row.components(separatedBy: ",")
                let vals = columns.compactMap { Double($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
                return vals.count >= 13 ? vals : nil
            }
        } catch { return [] }
    }

    private func getNormalizedWinner(sumProbs: [Double], classes: [String]) -> (label: String, conf: Double)? {
        let totalSum = sumProbs.reduce(0, +)
        guard totalSum > 0 else { return nil }
        let percentages = sumProbs.map { ($0 / totalSum) * 100 }
        if let idx = percentages.enumerated().max(by: { $0.element < $1.element })?.offset { return (classes[idx], percentages[idx]) }
        return nil
    }
}
