import Foundation

import CoreMotion

import SwiftUI

import Combine

import CoreML



// typealias MotionData = CMDeviceMotion



// *** Global Variable ***

public var GLOBAL_IS_RECORDING = false



// MARK: - Main ViewModel



class HARViewModel: ObservableObject {

    private var lastRecordedTimestamp: String = ""

    @Published var dualModeLabel: String = "No Prediction"

    private var currentLabelForLogging: String = "No Prediction"

    private var currentConfidenceForLogging: Double = 0.0

    

    // اضافه کردن این تابع به انتهای فایل HARViewModel.swift

    func getFinalCSVPath() -> URL? {

        // ۱. هدر ۱۶ ستونه دقیق

        let header = "roll,pitch,yaw,timestamp,activity,gravX,gravY,gravZ,userAccX,userAccY,userAccZ,gyroX,gyroY,gyroZ,confidence,ground_truth\n"

                

                // بقیه کدهای این تابع بدون تغییر می‌مانند

                guard !csvData.isEmpty else { return nil }

                var finalContent = csvData

                if !finalContent.hasPrefix("roll") {

                    finalContent = header + finalContent

                }

        

        // ۴. ساخت فایل فیزیکی

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("SensorData_Final.csv")

        do {

            try finalContent.write(to: url, atomically: true, encoding: .utf8)

            return url

        } catch {

            print("❌ Error: \(error)")

            return nil

        }

    }

    

    

    



    // --- کدهای کاملاً جدید برای پردازش مستقل پوشه CSV ---



    // ساختار جدید و مستقل برای نتایج دکمه پروسس (شیت سوم)

    struct FolderProcessResult: Identifiable {

        let id = UUID()

        let fileName: String

        let modelName: String

        let dominantActivity: String

        let confidence: Double

    }



    // لیست نتایج و کنترل‌کننده شیت سوم

    @Published var folderResultsList: [FolderProcessResult] = []

    @Published var showFolderSheet: Bool = false

    

    // ۱. ساختار جدید برای نتایج نهایی ۳ فایل

    struct FolderResult: Identifiable {

        let id = UUID()

        let fileName: String

        let modelName: String

        let activity: String

        let confidence: Double // درصد اطمینان

    }



    // ۲. لیست نتایج و کنترل‌کننده شیت سوم

    @Published var folderResults: [FolderResult] = []

    @Published var showThirdSheet: Bool = false

    @Published var showFolderReportSheet: Bool = false

    

    func processOfflineCSVFolder() {

        // 1. Clear previous results

        self.batchComparisonResults.removeAll()

        

        // 2. Exact filenames WITHOUT .csv (Extension is handled by withExtension)

        let fileNames = [

            "MASTER_dws_2026-02-28_21-14-03",

            "MASTER_Up_2026-02-28_21-12-11",

            "MASTER_Walking_2026-02-28_21-16-57"

        ]

        

        for fileName in fileNames {

            // 3. Look for file in Root (subdirectory removed)

            if let fileURL = Bundle.main.url(forResource: fileName, withExtension: "csv") {

                

                do {

                    // 4. Read file content

                    let content = try String(contentsOf: fileURL)

                    let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }

                    

                    // 5. Trim logic: skip header + 150 rows at start, remove 150 at end

                    // Needs at least 1 header + 150 start + 150 end + 1 data row = 302 lines

                    if lines.count > 301 {

                        let header = lines[0]

                        let startIndex = 151

                        let endIndex = lines.count - 150

                        

                        let trimmedRows = Array(lines[startIndex..<endIndex])

                        let finalContent = ([header] + trimmedRows).joined(separator: "\n")

                        

                        // 6. Write to a temporary file for the models to process

                        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("\(fileName)_trimmed.csv")

                        try finalContent.write(to: tempURL, atomically: true, encoding: .utf8)

                        

                        // 7. Run inference on the trimmed file

                        let fileResults = BatchOrchestrator.shared.runFullComparison(fileURL: tempURL)

                        

                        // 8. Map results and update the source filename

                        let taggedResults = fileResults.map { res -> ModelWinningResult in

                            var updated = res

                            updated.sourceFile = fileName

                            return updated

                        }

                        

                        self.batchComparisonResults.append(contentsOf: taggedResults)

                        print("✅ Successfully processed: \(fileName)")

                        

                    } else {

                        print("⚠️ File \(fileName) too short to trim (Lines: \(lines.count))")

                    }

                    

                } catch {

                    print("❌ Error reading content for \(fileName): \(error)")

                }

                

            } else {

                // This error appears if the file is NOT in Root or Target Membership is off

                print("❌ File NOT FOUND in Root: \(fileName).csv")

            }

        }

        

        // 9. Show the comparison report sheet

        self.showFolderReportSheet = true

    }

    

    // در انتهای کلاس HARViewModel

    @Published var batchComparisonResults: [ModelWinningResult] = []

    @Published var showComparisonSheet: Bool = false



    func runBatchCommand(fileURL: URL) {

        // ۱. پاکسازی نتایج قبلی برای جلوگیری از تداخل

        self.batchComparisonResults.removeAll()

        

        if self.isDualMode {

            // حالت مقایسه‌ای: هر ۴ مدل روی فایل اجرا شوند

            self.batchComparisonResults = BatchOrchestrator.shared.runFullComparison(fileURL: fileURL)

        } else {

            // حالت تکی: فقط مدل انتخاب شده فعلی روی فایل اجرا شود

            let result = runSingleOfflineInference(fileURL: fileURL)

            self.batchComparisonResults = [result]

        }

        

        // ۲. چاپ نتایج در کنسول برای تایید نهایی (تمامی حروف کوچک)

        print("--- final winning class comparison table (batch) ---")

        for res in batchComparisonResults {

            print("\(res.modelName.lowercased()): \(res.winningClass.lowercased())")

        }

        

        // ۳. باز کردن مستقیم صفحه دوم (Sheet)

        self.showComparisonSheet = true

    }



    // تابع کمکی برای استخراج نتیجه تکی از فایل اکسل

    private func runSingleOfflineInference(fileURL: URL) -> ModelWinningResult {

        let name: String

        let label: String

        

        switch currentModel {

        case .randomForest:

            name = "rf-120 (ms)"

            label = CSVManagerRF120Url.loadDataAndRunInference(fileURL: fileURL)?.first?.activity ?? "unknown"

        case .uciRandomForest75, .uciRandomForest:

            name = "rf-75 (uci)"

            label = CSVManagerUCI75Url.loadDataAndRunInference(fileURL: fileURL)?.first?.activity ?? "unknown"

        case .deepLearning:

            name = "cnn-attention"

            label = CSVManagerUrl.loadDataAndRunInference(fileURL: fileURL)?.first?.activity ?? "unknown"

        }

        return ModelWinningResult(modelName: name, winningClass: label, confidence: 100.0)

     }

    

    

    

    @Published var isDualMode: Bool = false

    @Published var isExtraToggle: Bool = false

    

    // Memory to store the latest label from each model for combined display

    private var latestLabels: [String: String] = ["RF120": "N/A", "UCI75": "N/A", "RF75": "N/A", "CNN": "N/A"]

    private var virtualTimestamp: Double = 0.0

    private var isReplaying = false

    private var lastRFLabel: String = "N/A"

    private let simulationService = SimulationService()

    

    // uciEngine removed completely to clean up dead code

    private let rf75Engine = RF75InferenceEngine()



    func testWithCSVFile() {

        self.isReplaying = true

        self.virtualTimestamp = 0.0

        

        SensorProvider.shared.stop()

        simulationService.stopSimulation()

        

        self.smartReport.removeAll()

        self.rawHistory.removeAll()

        self.probabilities = Array(repeating: 0.0, count: self.uciActivities.count)

        self.activityLabel = "Ready"

        self.isRecording = false

        

        simulationService.delegate = self

        self.isRecording = true

        self.startNewRecording()

        

        NotificationCenter.default.removeObserver(self, name: NSNotification.Name("SimulationFinished"), object: nil)

        NotificationCenter.default.addObserver(forName: NSNotification.Name("SimulationFinished"), object: nil, queue: .main) { [weak self] _ in

            self?.isRecording = false

            self?.isReplaying = false

            self?.finishRecording()

        }

        

        let msg = simulationService.startSimulation(fileName: "test_data")

        self.debugInfo = msg

    }

    

    @Published var activityManager = ActivityStateManager()

    

    // UI State

    @Published var confidenceValue: Double = 0.0

    @Published var latencyMessage: String = ""

    @Published var showLatencyPopup: Bool = false

    private var actionStartTime: Date?

    private var isWaitingForFirstPrediction: Bool = false

    

    // Recording State

     var csvData = ""

    private var rawHistory: [(time: Double, label: String)] = []

    

    @Published var activityLabel: String = "Ready"

    @Published var confidence: Double = 0.0

    

    @Published var isActive: Bool = false

    @Published var isRecording: Bool = false

    @Published var isAuditMode: Bool = false

    

    @Published var lastExportURL: URL?

    @Published var liveCounter: Int = 0

    @Published var recordedCount: Int = 0

    @Published var debugInfo: String = "Waiting..."

    @Published var probabilities: [Double] = [0, 0, 0, 0, 0, 0]

    

    // Smart Report

    @Published var smartReport: [ProcessedSegment] = []

    @Published var showReportSheet: Bool = false

    

    // Model Selection

    @Published var currentModel: ModelType = .randomForest

    

    private let cnnLabels = ["Jogging", "Sitting", "Standing", "Walking", "Walking Downstairs", "Walking Upstairs"]

    private let rfLabels = ["dws", "jog", "sit", "std", "ups", "wlk"]

    private let uciRFLabels = ["DOWNSTAIRS", "LAYING", "SITTING", "STANDING", "UPSTAIRS", "WALKING"]

    private let rf75Labels = ["JOG", "WALK", "UPSTAIRS", "DOWNSTAIRS", "SIT", "STAND", "LAY"]

    @Published var uciActivities: [String] = []

    

    init() {

        SensorProvider.shared.delegate = self

        self.isAuditMode = GLOBAL_IS_RECORDING

        self.isRecording = GLOBAL_IS_RECORDING

        self.uciActivities = rfLabels

        self.activityManager.updateModelType(isCNN: false)

    }

    

    func changeModel(to model: ModelType) {

        guard !isRecording else { return }

        self.currentModel = model

        

        switch model {

        case .randomForest:

            self.uciActivities = rfLabels

            self.activityManager.updateModelType(isCNN: false)

        case .uciRandomForest:



            self.uciActivities = rf75Labels

            self.activityManager.updateModelType(isCNN: false)

        case .uciRandomForest75:

            self.uciActivities = rf75Labels

            self.activityManager.updateModelType(isCNN: false)

        case .deepLearning:

            self.uciActivities = cnnLabels

            self.activityManager.updateModelType(isCNN: true)

        }

        

        self.probabilities = Array(repeating: 0.0, count: self.uciActivities.count)

        self.activityLabel = "Model: \(model.rawValue)"

    }

    

    func toggleSystem() {

            isActive.toggle()

            updateSensorState()

            

            let logger = SensorProvider.shared.dataLogger

            

            if isActive {

                

                    activityLabel = "Detecting..."

                    debugInfo = "Live View ON"

                    self.actionStartTime = Date()

                    self.isWaitingForFirstPrediction = true

                    

                    // Code to be added or modified:

                // Code to be added or modified:

                                    self.rawHistory.removeAll() // Clear previous prediction history from memory

                                    self.csvData = ""           // Reset the CSV string

                                    self.recordedCount = 0      // Reset the counter

                self.smartReport.removeAll()

                                                        self.batchComparisonResults.removeAll()

                                                        self.virtualTimestamp = 0.0 // Reset the timer base for new recordings

                

                

                // 2. Auto-start the 15-channel background recording

                if !logger.isRecording {

                    logger.startRecording(activityName: self.activityManager.selectedGroundTruth)

                }

                

            } else {

                activityLabel = "Paused"

                debugInfo = "Paused"

                showLatencyPopup = false

                

                if logger.isRecording {

                    // 3. Stop recording the CSV file

                    logger.stopRecording()

                    

                    // 4. Slight delay to ensure the OS completely saves the file to the disk

                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in

                        guard let self = self else { return }

                        

                        if let fileURL = logger.lastCapturedURL {

                            var offlineResults: [ProcessedSegment]? = nil

                            

                            // 5. Smart Routing: Check which model is currently active in the UI

                            if self.currentModel == .randomForest {

                                // Route to the RF-120 processor

                                offlineResults = CSVManagerRF120Url.loadDataAndRunInference(fileURL: fileURL)

                            } else if self.currentModel == .uciRandomForest {

                                // Route to the RF-75 processor (Temporary fallback for old UI)

                                offlineResults = CSVManagerUCI75Url.loadDataAndRunInference(fileURL: fileURL)

                                

                            } else if self.currentModel == .uciRandomForest75 {

                                // Route to the RF-75 processor

                                offlineResults = CSVManagerRF75Url.loadDataAndRunInference(fileURL: fileURL)

                                

                            } else {

                                // Default route to the CNN processor

                                offlineResults = CSVManagerUrl.loadDataAndRunInference(fileURL: fileURL)

                            }

                            

                            // 6. Display the results in the Report Sheet

                            if let results = offlineResults {

                                self.smartReport = results

                                self.showReportSheet = true

                            } else {

                                self.debugInfo = "Error: Data invalid or too short."

                            }

                        }

                    }

                }

            }

        }

    

    func toggleAuditMode() {

        isRecording.toggle()

        GLOBAL_IS_RECORDING = isRecording

        self.isAuditMode = isRecording

        

        if isRecording {

            startNewRecording()

        } else {

            finishRecording()

        }

        updateSensorState()

    }

    

    private func updateSensorState() {

        if isActive || isRecording {

            SensorProvider.shared.start()

        } else {

            SensorProvider.shared.stop()

            activityLabel = "Stopped"

            confidence = 0.0

        }

    }

    

    private func startNewRecording() {

            // --- FULL RAM & TIME RESET ---

            self.csvData = ""

            self.recordedCount = 0

            self.rawHistory.removeAll()

            self.smartReport.removeAll()

            self.batchComparisonResults.removeAll()

            self.lastExportURL = nil

            self.virtualTimestamp = 0.0

            self.actionStartTime = Date()

            // -----------------------------

            

            debugInfo = "🔴 Recording..."

            

            // 🟢 این یک خط را جایگزین تمام شرط‌های قبلیِ csvData = ... کنید

            csvData = "roll,pitch,yaw,timestamp,activity,gravX,gravY,gravZ,userAccX,userAccY,userAccZ,gyroX,gyroY,gyroZ,confidence,ground_truth\n"

        }



    private func finishRecording() {

        processSmartSequence()

        appendSmartReportToCSV()

        saveCSVFile()

        debugInfo = "💾 Saved."

        showReportSheet = true

    }

    

    private func processSmartSequence() {

            guard !rawHistory.isEmpty else { return }

            

            // -------------------------------------------------------------

            // ۱. اضافه شدن منطق Trim: حذف ۳ ثانیه از ابتدا و ۳ ثانیه از انتها

            // -------------------------------------------------------------

            var filteredHistory = rawHistory.sorted { $0.time < $1.time }

            if let firstTime = filteredHistory.first?.time, let lastTime = filteredHistory.last?.time {

                let trimDuration: TimeInterval = 3.0 // ۳ ثانیه

                let totalDuration = lastTime - firstTime

                

                // فقط در صورتی که زمان کل ضبط بیشتر از دو برابر زمان Trim (یعنی ۶ ثانیه) باشد، برش می‌دهیم

                if totalDuration > (trimDuration * 2) {

                    filteredHistory = filteredHistory.filter { entry in

                        let isAfterStart = (entry.time - firstTime) >= trimDuration

                        let isBeforeEnd = (lastTime - entry.time) >= trimDuration

                        return isAfterStart && isBeforeEnd

                    }

                    print("Trimmed 3s from start/end. Remaining points: \(filteredHistory.count)")

                } else {

                    print("Recording too short for trimming (\(totalDuration)s).")

                }

            }

            guard !filteredHistory.isEmpty else { return }

            

            var segments: [ProcessedSegment] = []

            var currentBlockSamples: [String] = []

            

            // ۲. استفاده از filteredHistory به جای rawHistory

            var blockStartTime = filteredHistory.first?.time ?? 0.0

            var currentStableState = filteredHistory.first?.label ?? ""

            let lookAheadCount = 2

            

            for i in 0..<filteredHistory.count {

                let item = filteredHistory[i]

                currentBlockSamples.append(item.label)

                

                if item.label != currentStableState {

                    // باید تابع isStateReallyChanged هم از filteredHistory استفاده کند (در ادامه اصلاح می‌شود)

                    if isStateReallyChanged(historyArray: filteredHistory, currentIndex: i, currentStableState: currentStableState, lookAhead: lookAheadCount) {

                        let endTime = item.time

                        let duration = endTime - blockStartTime

                        if duration > 0.1 {

                            let winnerLabel = getMajorityLabel(from: currentBlockSamples)

                            let type = isStatic(winnerLabel) ? "Static" : "Dynamic"

                            segments.append(ProcessedSegment(activity: winnerLabel, startTime: blockStartTime, duration: duration, type: type))

                        }

                        currentBlockSamples = []

                        blockStartTime = item.time

                        currentStableState = item.label

                    }

                }

            }

            

            if !currentBlockSamples.isEmpty {

                let endTime = filteredHistory.last?.time ?? blockStartTime

                let duration = endTime - blockStartTime

                if duration > 0.1 {

                    let winnerLabel = getMajorityLabel(from: currentBlockSamples)

                    let type = isStatic(winnerLabel) ? "Static" : "Dynamic"

                    segments.append(ProcessedSegment(activity: winnerLabel, startTime: blockStartTime, duration: duration, type: type))

                }

            }

            self.smartReport = segments

        }

    

    private func isStateReallyChanged(historyArray: [(time: Double, label: String)], currentIndex: Int, currentStableState: String, lookAhead: Int) -> Bool {

            if currentIndex + lookAhead >= historyArray.count { return true }

            var diffCount = 0

            for k in 1...lookAhead {

                if historyArray[currentIndex + k].label != currentStableState {

                    diffCount += 1

                }

            }

            return diffCount >= (lookAhead / 2)

        }

    

    private func isStatic(_ label: String) -> Bool {

        let l = label.lowercased()

        return l.contains("sit") || l.contains("stand") || l.contains("lay") || l.contains("std")

    }



    private func getMajorityLabel(from labels: [String]) -> String {

        guard !labels.isEmpty else { return "Unknown" }

        let counts = labels.reduce(into: [:]) { counts, label in counts[label, default: 0] += 1 }

        return counts.max(by: { $0.value < $1.value })?.key ?? labels[0]

    }

    

    private func appendSmartReportToCSV() {

        csvData.append("\n\n--- SMART SEQUENCE REPORT ---\n")

        csvData.append("Order,Phase,Activity,Duration,Start Time\n")

        

        for (index, item) in smartReport.enumerated() {

            csvData.append("\(index + 1),\(item.type),\(item.activity),\(String(format: "%.2f", item.duration)),\(String(format: "%.2f", item.startTime))\n")

        }

    }

    

    private func saveCSVFile() {

        let fileName = "HAR_Seq_\(currentModel == .randomForest ? "RF" : "CNN")_\(Int(Date().timeIntervalSince1970)).csv"

        let fileURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)

        

        do {

            try csvData.write(to: fileURL, atomically: true, encoding: .utf8)

            DispatchQueue.main.async {

                self.lastExportURL = fileURL

            }

        } catch {

            print("Error saving: \(error)")

        }

    }

    

    private func getIndex(for label: String) -> Int {

        let l = label.lowercased().prefix(1).uppercased() + label.lowercased().dropFirst()

        return uciActivities.firstIndex(of: l) ?? 0

    }

    

    private func forceLabel(label: String, index: Int) {

        self.activityLabel = label

        self.confidence = 100.0

        var probs = [Double](repeating: 0.0, count: uciActivities.count)

        if index < probs.count {

            probs[index] = 1.0

        }

        self.probabilities = probs

    }

    

    private func updateUI(with result: RecognitionResult) {

        self.activityLabel = result.label.uppercased()

        self.confidence = result.confidence

        self.probabilities = self.uciActivities.map { result.allProbs[$0] ?? 0.0 }

    }

    

    private func calculateTurbulence(accX: [Double], accY: [Double], accZ: [Double]) -> Double {

        var magnitudes = [Double]()

        let count = min(accX.count, accY.count, accZ.count)

        for i in 0..<count {

            let m = sqrt(pow(accX[i], 2) + pow(accY[i], 2) + pow(accZ[i], 2))

            magnitudes.append(m)

        }

        

        let mean = magnitudes.reduce(0, +) / Double(count)

        let variance = magnitudes.reduce(0) { $0 + pow($1 - mean, 2) } / Double(count)

        return sqrt(variance)

    }

    

    private func checkLatency() {

        if isWaitingForFirstPrediction, let startTime = actionStartTime {

            let timeDiff = Date().timeIntervalSince(startTime)

            self.latencyMessage = String(format: "Latency: %.3fs", timeDiff)

            

            withAnimation {

                self.showLatencyPopup = true

            }

            

            self.isWaitingForFirstPrediction = false

            

            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {

                withAnimation {

                    self.showLatencyPopup = false

                }

            }

        }

    }



    // Helper to update the combined UI label for Dual and Extra modes independently

    private func updateCombinedLabel(currentLabel: String) {

        if isDualMode {

            self.activityLabel = "RF120: \(latestLabels["RF120"]!) | UCI75: \(latestLabels["UCI75"]!) | RF75: \(latestLabels["RF75"]!) | CNN: \(latestLabels["CNN"]!)"

        } else if isExtraToggle {

            self.activityLabel = "UCI75: \(latestLabels["UCI75"]!) | RF75: \(latestLabels["RF75"]!)"

        } else {

            self.activityLabel = currentLabel

        }

    }

}



// MARK: - Delegate Processing



extension HARViewModel: SensorProviderDelegate {

    

    func didCollectFullWindow(

            rolls: [Double], pitches: [Double], yaws: [Double],

            userAccX: [Double], userAccY: [Double], userAccZ: [Double],

            gravX: [Double], gravY: [Double], gravZ: [Double],

            gyroX: [Double], gyroY: [Double], gyroZ: [Double],

            rawAccX: [Double], rawAccY: [Double], rawAccZ: [Double]

        ) {

            if self.isRecording {

                        let baseTime = self.isReplaying ? self.virtualTimestamp : Date().timeIntervalSince1970

                        let labelToWrite = self.isDualMode ? self.dualModeLabel : self.activityLabel

                        let groundTruth = self.activityManager.selectedGroundTruth

                        

                        var windowBlock = ""

                        

                        for i in 0..<userAccX.count {

                            let sampleTime = baseTime + (Double(i) * 0.02)

                            

                            // 🟢 چیدمان دقیقاً مطابق انتظار BatchOrchestrator

                            let line = String(format: "%.4f,%.4f,%.4f,%.3f,%@,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%@\n",

                                              rolls[i], pitches[i], yaws[i],           // ایندکس 0, 1, 2

                                              sampleTime,                              // ایندکس 3

                                              labelToWrite,                            // (حذف می‌شود در پردازش)

                                              gravX[i], gravY[i], gravZ[i],            // ایندکس 4, 5, 6 (محل دقیق جاذبه)

                                              userAccX[i], userAccY[i], userAccZ[i],   // ایندکس 7, 8, 9

                                              gyroX[i], gyroY[i], gyroZ[i],            // ایندکس 10, 11, 12

                                              self.confidence,                         // ایندکس 13

                                              groundTruth                              // (حذف می‌شود)

                            )

                            windowBlock.append(line)

                        }

                        

                        self.csvData.append(windowBlock)

                        self.recordedCount += 1

                    }

            

            if currentModel == .randomForest || isDualMode {

                

                let features = FeatureExtractor.extractFeatures(

                    rolls: rolls, pitches: pitches, yaws: yaws,

                    gravX: gravX, gravY: gravY, gravZ: gravZ,

                    rotX: gyroX, rotY: gyroY, rotZ: gyroZ,

                    userAccX: userAccX, userAccY: userAccY, userAccZ: userAccZ

                )

                

                DispatchQueue.global(qos: .userInitiated).async {

                    do {

                        let mlInput = try MLMultiArray(shape: [120], dataType: .double)

                        for (i, val) in features.enumerated() {

                            mlInput[i] = NSNumber(value: val)

                        }

                        

                        let model = try MotionSense_120(configuration: MLModelConfiguration())

                        let input = MotionSense_120Input(features: mlInput)

                        let output = try model.prediction(input: input)

                        

                        DispatchQueue.main.async {

                            var rawProbs = self.rfLabels.map { output.classProbability[$0] ?? 0.0 }

                            let totalSum = rawProbs.reduce(0, +)

                            if totalSum > 0 {

                                rawProbs = rawProbs.map { $0 / totalSum }

                            }

                            

                            let maxProb = rawProbs.max() ?? 0.0

                            let finalConf = maxProb * 100.0

                            let finalLabel = output.classLabel

                            

                            self.checkLatency()

                            self.latestLabels["RF120"] = finalLabel

                            self.probabilities = rawProbs

                            self.confidence = finalConf

                            self.lastRFLabel = finalLabel

                            self.updateCombinedLabel(currentLabel: finalLabel)

                            

                            if self.isActive {

                                self.debugInfo = "Live RF: \(finalLabel)"

                            } else {

                                self.debugInfo = "Replay: \(finalLabel)"

                            }

                        }

                    } catch {

                        print("❌ RF Error: \(error)")

                    }

                }

            }

            

            // 1.5 NEW HYBRID RANDOM FOREST LOGIC (75 Features)

            if currentModel == .uciRandomForest75 || currentModel == .uciRandomForest || isDualMode || isExtraToggle {

                rf75Engine.runInference(

                    userAccX: userAccX, userAccY: userAccY, userAccZ: userAccZ,

                    rawAccX: rawAccX, rawAccY: rawAccY, rawAccZ: rawAccZ,

                    gyroX: gyroX, gyroY: gyroY, gyroZ: gyroZ

                ) { label, rawConfidence, probsDict in

                    

                    DispatchQueue.main.async {

                        self.checkLatency()

                        

                        var rawProbs = self.uciActivities.map { probsDict[$0] ?? 0.0 }

                        let totalSum = rawProbs.reduce(0, +)

                        

                        if totalSum > 0 {

                            rawProbs = rawProbs.map { $0 / totalSum }

                        } else {

                            rawProbs = Array(repeating: 0.0, count: self.uciActivities.count)

                            if let index = self.uciActivities.firstIndex(of: label) { rawProbs[index] = 1.0 }

                        }

                        

                        self.latestLabels["RF75"] = label

                 

                        self.latestLabels["UCI75"] = label

                        self.probabilities = rawProbs

                        let maxProb = rawProbs.max() ?? 0.0

                        self.confidence = maxProb * 100.0

                        self.updateCombinedLabel(currentLabel: label)

                        

                    }

                }

            }

            

            // 2. CNN LOGIC

            if currentModel == .deepLearning || isDualMode {

                DispatchQueue.global(qos: .userInitiated).async { [weak self] in

                    guard let self = self else { return }

                    

                    let cleanedWindow = PreProcessor.shared.cleanWindow(

                        userAccX: userAccX, userAccY: userAccY, userAccZ: userAccZ,

                        gyroX: gyroX, gyroY: gyroY, gyroZ: gyroZ,

                        totalAccX: rawAccX, totalAccY: rawAccY, totalAccZ: rawAccZ

                    )

                    

                    let nnResult = InferenceEngine.shared.predict(cleanedWindow: cleanedWindow)

                    

                    DispatchQueue.main.async {

                        self.liveCounter += 1

                        var finalLabel = "Unknown"

                        var finalConf = 0.0

                        self.checkLatency()

                        

                        if let result = nnResult {

                            let turbulence = self.calculateTurbulence(accX: rawAccX, accY: rawAccY, accZ: rawAccZ)

                            let zVal = abs(rawAccZ.last ?? 0.0)

                            let xVal = abs(rawAccX.last ?? 0.0)

                            let yVal = abs(rawAccY.last ?? 0.0)

                            

                            if turbulence < 0.2 && (zVal > xVal && zVal > yVal) {

                                finalLabel = "LAYING"

                                finalConf = 100.0

                                self.activityLabel = finalLabel

                                self.confidence = finalConf

                                self.probabilities = Array(repeating: 0.0, count: self.uciActivities.count)

                            } else {

                                finalLabel = result.label

                                finalConf = result.confidence

                                self.updateUI(with: result)

                            }

                            

                            self.latestLabels["CNN"] = finalLabel

                            self.confidence = finalConf

                            self.updateCombinedLabel(currentLabel: finalLabel)

                        }

                        

                        if !self.isActive {

                            self.debugInfo = "Replay: \(finalLabel)"

                        }



                        // Combined history management

                        let historyLabel = self.activityLabel

                        if !self.isDualMode && !self.isExtraToggle {

                            if self.isReplaying { self.virtualTimestamp += 1.28 }

                        }

                        let t = self.isReplaying ? self.virtualTimestamp : Date().timeIntervalSince1970

                        

                        self.rawHistory.append((time: t, label: historyLabel))

                        

  // Ground_Truth

                            

                   

                        

                    }

                }

            }

        }

}