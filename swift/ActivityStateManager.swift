//
//  Untitled.swift
//  har2
//
//  Created by BEM on 19/02/2026.
//
import Foundation
import CoreMotion
import SwiftUI
import Combine
import CoreML

typealias MotionData = CMDeviceMotion

class ActivityStateManager: ObservableObject {
    private let cnnClasses = [
        "Walking", "Walking Upstairs", "Walking Downstairs",
        "Sitting", "Standing", "Laying", "Running"
    ]
    
    private let rfClasses = [
        "wlk", "ups", "dws", "sit", "std", "jog"
    ]
    
    @Published var currentClassList: [String] = []
    @Published var selectedGroundTruth: String = ""
    
    init() {
        updateModelType(isCNN: false)
    }
    
    func updateModelType(isCNN: Bool) {
        if isCNN {
            currentClassList = cnnClasses
            if !cnnClasses.contains(selectedGroundTruth) {
                selectedGroundTruth = cnnClasses.first ?? "Walking"
            }
        } else {
            currentClassList = rfClasses
            if !rfClasses.contains(selectedGroundTruth) {
                selectedGroundTruth = rfClasses.first ?? "wlk"
            }
        }
    }
}
