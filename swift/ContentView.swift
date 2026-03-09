import SwiftUI





struct FolderComparisonView: View {
    
    
    
    @State private var processedShareURL: URL? = nil
    let results: [HARViewModel.FolderResult]
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            List {
                // دسته‌بندی نتایج بر اساس نام فایل
                let groupedResults = Dictionary(grouping: results, by: { $0.fileName })
                
                ForEach(groupedResults.keys.sorted(), id: \.self) { fileName in
                    Section(header: Text("FILE: \(fileName)").font(.headline).foregroundColor(.purple)) {
                        ForEach(groupedResults[fileName] ?? []) { res in
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(res.modelName)
                                        .font(.caption)
                                        .fontWeight(.bold)
                                        .foregroundColor(.gray)
                                    Text(res.activity.uppercased())
                                        .font(.headline)
                                        .foregroundColor(.primary)
                                }
                                Spacer()
                                // نمایش درصد اطمینان
                                Text(String(format: "%.1f%%", res.confidence))
                                    .font(.system(.body, design: .monospaced))
                                    .padding(6)
                                    .background(Color.purple.opacity(0.1))
                                    .cornerRadius(6)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Folder Process Report")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

struct ContentView: View {
    @StateObject private var viewModel = HARViewModel()
    @ObservedObject private var dataLogger = SensorProvider.shared.dataLogger
    @State private var selectedModel: ModelType = .randomForest
    @State private var showShareSheet = false

    var body: some View {
        ZStack {
            Color(UIColor.systemBackground).ignoresSafeArea()
            
            VStack(spacing: 0) {
                Spacer().frame(height: 15)
                
                // ۱. دکمه‌های رادیویی انتخاب مدل
                modelSelector
                
                // ۲. کادر وضعیت زرد
                debugStatusBar
                
                ScrollView {
                    VStack(spacing: 15) {
                        // ۳. نمایش فعالیت و اطمینان
                        statusDisplay
                        
                        // ۴. نمودار میله‌ای
                        if !viewModel.probabilities.isEmpty {
                            probabilityChart
                        }
                        
                        // ۵. انتخابگر Ground Truth
                        groundTruthPicker
                        
                        // ۶. پنل دکمه‌های کنترلی
                        controlButtons
                    }
                    .padding(.horizontal)
                    .padding(.bottom, 20)
                }
            }
            
            // نمایش پاپ‌آپ Latency
            if viewModel.showLatencyPopup {
                latencyPopup
            }
        }
        // --- بخش شیت‌های به‌روزرسانی شده ---
        .sheet(isPresented: $viewModel.showReportSheet) {
            SessionReportView(report: viewModel.smartReport) { viewModel.showReportSheet = false }
        }
        // ✅ اصلاح شده: شیت نمایش نتایج مقایسه‌ای هر 4 مدل با ارسال وضعیت Dual
        .sheet(isPresented: $viewModel.showComparisonSheet) {
            ComparisonResultView(
                results: viewModel.batchComparisonResults,
                isDualMode: viewModel.isDualMode
            )
        }
        .sheet(isPresented: $showShareSheet) {
            shareSheetContent
        }.sheet(isPresented: $viewModel.showFolderReportSheet) {
            // این شیت سوم است که نتایج ۳ فایل پوشه را نشان می‌دهد
            ComparisonResultView(
                results: viewModel.batchComparisonResults,
                isDualMode: true
            )
        }
        // --------------------------------
        .onAppear {
            viewModel.changeModel(to: selectedModel)
            let verifier = UCIVerifier()
            verifier.runAudit()
            
            let tester = CoreMLParityTester()
            tester.runParityTest()
            
            let verifierv = ModelVerifierv()
            verifierv.runVerification()
            
            CSVManager.loadDataAndRunInference(fileName: "MASTER_Walking_2026-02-28_21-16-57")
        }
    }

    // MARK: - Subviews (بدون تغییر در منطق قبلی)

    private var modelSelector: some View {
        HStack(spacing: 10) {
            ForEach(ModelType.allCases) { model in
                Button(action: {
                    withAnimation(.spring()) {
                        selectedModel = model
                        viewModel.changeModel(to: model)
                    }
                }) {
                    VStack(spacing: 2) {
                        Text(model == .uciRandomForest75 ? "RF-Comp" : (model == .uciRandomForest ? "UCI" : (model == .randomForest ? "RF-OLD" : "CNN")))
                            .font(.system(size: 14, weight: .black))
                        Text(model == .uciRandomForest75 ? "75-Feat" : (model == .uciRandomForest ? "75-Feat" : (model == .randomForest ? "120-Feat" : "Deep")))
                            .font(.system(size: 8))
                    }
                    .foregroundColor(selectedModel == model ? .white : .primary)
                    .frame(maxWidth: .infinity).frame(height: 45)
                    .background(selectedModel == model ? getModelColor(model) : Color(UIColor.secondarySystemBackground))
                    .cornerRadius(12)
                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(selectedModel == model ? getModelColor(model) : Color.gray.opacity(0.2), lineWidth: 1.5))
                }
            }
        }
        .padding(.horizontal).padding(.bottom, 10)
    }

    private var debugStatusBar: some View {
        Text(viewModel.debugInfo.isEmpty ? "SYSTEM READY" : viewModel.debugInfo)
            .font(.system(size: 11, weight: .bold))
            .padding(8).frame(maxWidth: .infinity)
            .background(Color.yellow).foregroundColor(.black).cornerRadius(8).padding(.horizontal)
    }

    private var statusDisplay: some View {
        VStack(spacing: 5) {
            Text(viewModel.activityLabel)
                .font(.system(size: 40, weight: .black))
                .foregroundColor(getStatusColor()).minimumScaleFactor(0.5)
            
            Text(String(format: "CONFIDENCE: %.1f%%", min(max(viewModel.confidence, 0.0), 100.0)))
                .font(.headline).foregroundColor(.green).opacity(viewModel.isActive ? 1 : 0)
        }
        .padding(.top, 10)
    }

    private var probabilityChart: some View {
        HStack(alignment: .bottom, spacing: 8) {
            ForEach(0..<viewModel.probabilities.count, id: \.self) { index in
                VStack {
                    let val = viewModel.probabilities[index]
                    RoundedRectangle(cornerRadius: 4)
                        .fill(barColor(for: index))
                        .frame(width: 35, height: CGFloat(min(max(val, 0), 1) * 140 + 5))
                    Text(getActivityShortName(index: index))
                        .font(.system(size: 9, weight: .bold)).foregroundColor(.gray).rotationEffect(.degrees(-45))
                }
            }
        }
        .frame(height: 170).padding(.bottom, 10)
    }

    private var groundTruthPicker: some View {
        VStack(spacing: 2) {
            Text("ACTUAL ACTIVITY (Ground Truth)").font(.caption2).fontWeight(.bold).foregroundColor(.gray)
            Picker("Ground Truth", selection: $viewModel.activityManager.selectedGroundTruth) {
                ForEach(viewModel.activityManager.currentClassList, id: \.self) { activity in
                    Text(activity).tag(activity)
                }
            }
            .pickerStyle(WheelPickerStyle()).frame(height: 90)
            .background(Color(UIColor.secondarySystemBackground)).cornerRadius(10)
        }
    }

    private var controlButtons: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                // ۱. دکمه لغزنده Dual Mode
                VStack {
                    Text("Dual").font(.system(size: 10, weight: .bold))
                    Toggle("", isOn: $viewModel.isDualMode)
                        .labelsHidden()
                        .toggleStyle(SwitchToggleStyle(tint: .orange))
                }.frame(width: 60)

                VStack {
                    Text("process").font(.system(size: 10, weight: .bold))
                    Button(action: {
                        // فراخوانی تابع جدید برای پردازش پوشه آفلاین
                        viewModel.processOfflineCSVFolder()
                    }) {
                        Image(systemName: "play.circle.fill")
                            .font(.system(size: 24))
                            // دکمه را همیشه بنفش نگه می‌داریم تا برای پوشه آفلاین آماده باشد
                            .foregroundColor(.purple)
                    }
                }.frame(width: 60)
                
                Spacer()

                // ۳. دکمه استارت لحظه‌ای
                Button(action: { viewModel.toggleSystem() }) {
                    Text(viewModel.isActive ? "STOP" : "START")
                        .font(.system(size: 14, weight: .bold))
                        .foregroundColor(.white)
                        .frame(width: 150, height: 44)
                        .background(viewModel.isActive ? Color.red : Color.green)
                        .cornerRadius(10)
                }
            }
            .padding(.horizontal)

            Button(action: { viewModel.toggleAuditMode() }) {
                Label(viewModel.isAuditMode ? "STOP REPORT" : "START REPORT", systemImage: viewModel.isAuditMode ? "stop.circle.fill" : "doc.text.fill")
                    .bold().foregroundColor(.white).frame(maxWidth: .infinity).frame(height: 50)
                    .background(viewModel.isAuditMode ? Color.red : Color.blue).cornerRadius(12)
            }

            Button(action: {
                if dataLogger.isRecording { dataLogger.stopRecording() }
                else { dataLogger.startRecording(activityName: viewModel.activityManager.selectedGroundTruth) }
            }) {
                Label(dataLogger.isRecording ? "STOP MASTER" : "START MASTER (15-COL)", systemImage: dataLogger.isRecording ? "stop.circle.fill" : "record.circle")
                    .bold().foregroundColor(.white).frame(maxWidth: .infinity).frame(height: 50)
                    .background(dataLogger.isRecording ? Color.red : Color.indigo).cornerRadius(12)
            }

            HStack(spacing: 10) {
                Button(action: { if hasFileToShare() { showShareSheet = true } }) {
                    Label(getShareButtonText(), systemImage: "square.and.arrow.up")
                        .font(.system(size: 11, weight: .bold)).foregroundColor(.white)
                        .frame(maxWidth: .infinity).frame(height: 50)
                        .background(hasFileToShare() ? Color.orange : Color.gray.opacity(0.4)).cornerRadius(12)
                }.disabled(!hasFileToShare())

                Button(action: { exit(0) }) {
                    Label("EXIT", systemImage: "power").font(.system(size: 11, weight: .bold))
                        .foregroundColor(.white).frame(maxWidth: .infinity).frame(height: 50)
                        .background(Color.red).cornerRadius(12)
                }
            }
        }
    }

    private var latencyPopup: some View {
        VStack {
            Text(viewModel.latencyMessage).font(.system(size: 14, weight: .bold)).foregroundColor(.white)
                .padding(.horizontal, 20).padding(.vertical, 10)
                .background(Capsule().fill(Color.blue.opacity(0.9))).padding(.top, 50)
            Spacer()
        }.zIndex(2).transition(.asymmetric(insertion: .move(edge: .top).combined(with: .opacity), removal: .opacity))
    }

    private var shareSheetContent: some View {
        if let masterURL = dataLogger.lastCapturedURL { return AnyView(ActivityView(activityItems: [masterURL])) }
        else if let oldURL = viewModel.lastExportURL { return AnyView(ActivityView(activityItems: [oldURL])) }
        else { return AnyView(Text("No File Found")) }
    }

    // MARK: - Helpers

    func getModelColor(_ model: ModelType) -> Color {
        switch model {
        case .uciRandomForest: return .blue
        case .randomForest: return .purple
        case .deepLearning: return .green
        case .uciRandomForest75: return .orange
        }
    }

    func hasFileToShare() -> Bool { return dataLogger.lastCapturedURL != nil || viewModel.lastExportURL != nil }
    
    func getShareButtonText() -> String {
        if dataLogger.lastCapturedURL != nil { return "SHARE MASTER" }
        else if viewModel.lastExportURL != nil { return "SHARE OLD" }
        else { return "NO FILE" }
    }

    func getStatusColor() -> Color {
        if dataLogger.isRecording { return .purple }
        if viewModel.isRecording { return .red }
        if !viewModel.isActive { return .secondary }
        if viewModel.activityLabel == viewModel.activityManager.selectedGroundTruth { return .green }
        return .primary
    }

    func barColor(for index: Int) -> Color {
        guard index < viewModel.uciActivities.count else { return .gray }
        let label = viewModel.uciActivities[index].uppercased()
        if ["RUNNING", "JOG"].contains(label) { return .purple }
        if ["WALKING", "WLK", "UPS", "DWS", "UPSTAIRS", "DOWNSTAIRS"].contains(label) { return .blue }
        return .orange
    }

    func getActivityShortName(index: Int) -> String {
        guard index < viewModel.uciActivities.count else { return "?" }
        let label = viewModel.uciActivities[index].uppercased()
        switch label {
        case "WALKING", "WLK": return "WLK"
        case "WALKINGUPSTAIRS", "UPS", "UPSTAIRS": return "UP"
        case "WALKINGDOWNSTAIRS", "DWS", "DOWNSTAIRS": return "DWN"
        case "SITTING", "SIT": return "SIT"
        case "STANDING", "STD": return "STD"
        case "LAYING", "LAY": return "LAY"
        case "RUNNING", "JOG": return "RUN"
        default: return String(label.prefix(3))
        }
    }
}

struct ComparisonResultView: View {
    let results: [ModelWinningResult]
    let isDualMode: Bool
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 25) {
                    // ۱. دسته‌بندی نتایج بر اساس نام فایل
                    let groupedResults = Dictionary(grouping: results, by: { $0.sourceFile })
                    
                    // ۲. مرتب‌سازی نام فایل‌ها و نمایش هر کدام در یک بخش جداگانه
                    ForEach(groupedResults.keys.sorted(), id: \.self) { fileName in
                        VStack(alignment: .leading, spacing: 15) {
                            
                            // هدر نام فایل (بنفش)
                            HStack {
                                Image(systemName: "doc.text.fill")
                                Text("FILE: \(fileName)")
                                    .font(.system(size: 14, weight: .black, design: .monospaced))
                            }
                            .foregroundColor(.purple)
                            .padding(.horizontal)

                            // نمایش مدل‌های مربوط به همین فایل خاص
                            // ... داخل بدنه ComparisonResultView و در بخش نمایش مدل‌ها ...

                            VStack(spacing: 10) {
                                // تبدیل آرایه به لیست شماره‌گذاری شده برای تشخیص ردیف‌های زوج و فرد
                                let models = groupedResults[fileName] ?? []
                                
                                ForEach(Array(models.enumerated()), id: \.element.id) { index, res in
                                    HStack {
                                        VStack(alignment: .leading, spacing: 4) {
                                            Text(res.modelName.lowercased())
                                                .font(.system(size: 11, weight: .bold, design: .monospaced))
                                                .foregroundColor(.gray)
                                            Text(res.winningClass.lowercased())
                                                .font(.system(size: 16, weight: .black))
                                                .foregroundColor(.blue)
                                        }
                                        Spacer()
                                        
                                        // تعیین رنگ فقط بر اساس ردیف (ردیف ۱ و ۳ سبز، ردیف ۲ و ۴ نارنجی)
                                        Text(String(format: "%.1f%%", res.confidence))
                                            .font(.system(size: 15, weight: .bold, design: .monospaced))
                                            .foregroundColor(index % 2 == 0 ? .green : .orange) // منطق یکی در میان
                                    }
                                    .padding()
                                    .background(RoundedRectangle(cornerRadius: 12).fill(Color.gray.opacity(0.08)))
                                    .padding(.horizontal)
                                }
                            }
                            Divider()
                                .background(Color.gray.opacity(0.3))
                                .padding(.top, 10)
                        }
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("batch analysis")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                Button("done") { dismiss() }
            }
        }
    }
}
