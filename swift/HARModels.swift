//
//  HARModels.swift
//  har2
//
//  Created by BEM on 19/02/2026.
//

import Foundation


struct ProcessedSegment: Identifiable {
    let id = UUID()
    let activity: String
    let startTime: Double
    let duration: Double
    let type: String
    var confidence: Double = 0.0 // این خط اضافه شد تا ارور ارکستریتور رفع شود
}




enum ModelType: String, CaseIterable, Identifiable {
    case deepLearning = "Deep Learning (CNN)"
    case randomForest = "MotionSense RF (120)"
     case uciRandomForest = "UCI Random Forest"
    case uciRandomForest75 = "hybrid Random Forest"
    var id: String { self.rawValue }
}
