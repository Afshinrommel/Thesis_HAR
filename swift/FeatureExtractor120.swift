import Foundation
import Accelerate

class FeatureExtractor {
    
    static func extractFeatures(
        rolls: [Double], pitches: [Double], yaws: [Double],
        gravX: [Double], gravY: [Double], gravZ: [Double],
        rotX: [Double], rotY: [Double], rotZ: [Double],
        userAccX: [Double], userAccY: [Double], userAccZ: [Double]
    ) -> [Double] {
        
        var features: [Double] = []
        
        // Exact order from Python: Attitude, Gravity, RotationRate, UserAcceleration
        let allChannels = [
            rolls, pitches, yaws,
            gravX, gravY, gravZ,
            rotX, rotY, rotZ,
            userAccX, userAccY, userAccZ // ⚠️ یادتان باشد: اینجا Y منفی نمی‌شود!
        ]
        
        for channelData in allChannels {
            let feats = calculateChannelFeatures(channelData)
            features.append(contentsOf: feats)
        }
        
        return features
    }
    
    private static func calculateChannelFeatures(_ data: [Double]) -> [Double] {
        // 1. Mean
        let mean = vDSP.mean(data)
        
        // 2. Std Dev (Population Std Dev to match np.std)
        let sqData = vDSP.square(data)
        let meanSq = vDSP.mean(sqData)
        let variance = meanSq - (mean * mean)
        let stdDev = sqrt(max(0.0, variance))
        
        // 3. Max, 4. Min
        let maxVal = data.max() ?? 0.0
        let minVal = data.min() ?? 0.0
        
        // 5. Median
        let median = calculateMedian(data)
        
        // 6. Mean Absolute Value
        let absData = vDSP.absolute(data)
        let meanAbs = vDSP.mean(absData)
        
        // 7. Mean Square (already calculated)
        
        // --- FFT Features ---
        let fftAmps = calculateFFT(data)
        
        // 8. ArgMax FFT
        let argMaxFFT = Double(fftAmps.indices.max(by: { fftAmps[$0] < fftAmps[$1] }) ?? 0)
        
        // 9. Mean FFT Square
        let fftSq = vDSP.square(fftAmps)
        let meanFFTSq = vDSP.mean(fftSq)
        
        // 10. Entropy
        let entropy = calculateEntropyPythonStyle(fftAmps)
        
        // Exact order appended in python:
        // mean, std, max, min, median, mean_abs, mean_sq, argmax_fft, mean_fft_sq, entropy
        return [mean, stdDev, maxVal, minVal, median, meanAbs, meanSq, argMaxFFT, meanFFTSq, entropy]
    }
    
    private static func calculateMedian(_ data: [Double]) -> Double {
        let sorted = data.sorted()
        if sorted.isEmpty { return 0 }
        let count = sorted.count
        if count % 2 == 1 {
            return sorted[count / 2]
        } else {
            return (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        }
    }
    
    // Replicating np.abs(fft(signal))[:n//2]
    private static func calculateFFT(_ data: [Double]) -> [Double] {
        let n = data.count
        let halfN = n / 2
        var amps = [Double]()
        
        for k in 0..<halfN {
            var sr = 0.0
            var si = 0.0
            for (t, val) in data.enumerated() {
                let angle = 2.0 * .pi * Double(t) * Double(k) / Double(n)
                sr += val * cos(angle)
                si += -val * sin(angle)
            }
            let mag = sqrt(sr * sr + si * si)
            amps.append(mag)
        }
        return amps
    }
    
    private static func calculateEntropyPythonStyle(_ fftAmps: [Double]) -> Double {
        let sumFFT = fftAmps.reduce(0, +)
        if sumFFT == 0 { return 0.0 }
        
        let epsilon = 1e-9
        let denominator = sumFFT + epsilon
        
        var entropy = 0.0
        for amp in fftAmps {
            let p = amp / denominator
            entropy += p * log2(p + epsilon)
        }
        return -entropy
    }
}

extension vDSP {
    static func mean(_ vector: [Double]) -> Double {
        var result = 0.0
        vDSP_meanvD(vector, 1, &result, vDSP_Length(vector.count))
        return result
    }
    static func absolute(_ vector: [Double]) -> [Double] {
        var result = [Double](repeating: 0.0, count: vector.count)
        vDSP_vabsD(vector, 1, &result, 1, vDSP_Length(vector.count))
        return result
    }
    static func square(_ vector: [Double]) -> [Double] {
        var result = [Double](repeating: 0.0, count: vector.count)
        vDSP_vsqD(vector, 1, &result, 1, vDSP_Length(vector.count))
        return result
    }
}
