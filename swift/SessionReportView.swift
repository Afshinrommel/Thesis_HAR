
import SwiftUI

struct SessionReportView: View {
    let report: [ProcessedSegment]
    let onClose: () -> Void
    var body: some View {
        NavigationView {
            List {
                if report.isEmpty { Text("No activity recorded.").foregroundColor(.secondary) }
                else {
                    Section(header: Text("Smart Sequence Analysis")) {
                        ForEach(Array(report.enumerated()), id: \.element.id) { index, item in
                            HStack(spacing: 15) {
                                ZStack {
                                    Circle().fill(Color.gray.opacity(0.2)).frame(width: 30, height: 30)
                                    Text("\(index + 1)").font(.caption).bold()
                                }
                                Image(systemName: item.type == "Static" ? "pause.circle.fill" : "figure.walk.circle.fill")
                                    .font(.title2).foregroundColor(colorFor(item.activity))
                                VStack(alignment: .leading) {
                                    Text(item.activity.uppercased()).font(.headline).bold().foregroundColor(colorFor(item.activity))
                                    Text(item.type).font(.caption).foregroundColor(.secondary)
                                }
                                Spacer()
                                Text(String(format: "%.1f s", item.duration)).font(.title3).bold()
                            }.padding(.vertical, 5)
                        }
                    }
                    
                    // ... بعد از بخش Smart Sequence Analysis در List ...

                    Section(header: Text("Top 3 Activities (Overall Summary)")) {
                        // محاسبه ۳ فعالیت برتر بر اساس مدت زمان یا تعداد تکرار
                        let summary = Dictionary(grouping: report, by: { $0.activity })
                            .mapValues { $0.reduce(0) { $0 + $1.duration } }
                            .sorted { $0.value > $1.value }
                            .prefix(3)
                        
                        let totalTime = report.reduce(0) { $0 + $1.duration }

                        ForEach(Array(summary.enumerated()), id: \.offset) { index, element in
                            HStack {
                                Text("\(index + 1). \(element.key.uppercased())")
                                    .bold()
                                    .foregroundColor(colorFor(element.key))
                                Spacer()
                                // نمایش درصد حضور هر فعالیت در کل فایل
                                Text(String(format: "%.1f%%", (element.value / totalTime) * 100))
                                    .fontWeight(.black)
                            }
                        }
                    }
                    Section {
                        let total = report.reduce(0) { $0 + $1.duration }
                        HStack { Text("Total Time"); Spacer(); Text(String(format: "%.1f s", total)).bold().foregroundColor(.green) }
                    }
                }
            }.navigationTitle("Activity Report")
            .toolbar { ToolbarItem(placement: .navigationBarTrailing) { Button(action: onClose) { Image(systemName: "xmark.circle.fill").font(.title).foregroundColor(.gray) } } }
        }
    }
    func colorFor(_ label: String) -> Color {
        let l = label.lowercased()
        if l.contains("walk") || l.contains("wlk") { return .blue }
        if l.contains("run") || l.contains("jog") { return .purple }
        if l.contains("sit") || l.contains("stand") || l.contains("std") || l.contains("lay") { return .orange }
        if l.contains("up") { return .green }
        if l.contains("dws") || l.contains("down") { return .red }
        return .gray
    }
}

struct ActivityView: UIViewControllerRepresentable {
    let activityItems: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController { UIActivityViewController(activityItems: activityItems, applicationActivities: nil) }
    func updateUIViewController(_ ui: UIActivityViewController, context: Context) {}
}
