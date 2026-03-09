import Foundation
import Accelerate
import CoreML

class FeatureExtractor75 {
    
    let featureCount = 75
    let windowSize = 128
    private let log2n = vDSP_Length(7)
    private let fftSetup: FFTSetupD

    init() {
        guard let setup = vDSP_create_fftsetupD(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("❌ FFT Setup Failed")
        }
        self.fftSetup = setup
    }

    deinit { vDSP_destroy_fftsetupD(fftSetup) }

    func extractFeatures(userAccX: [Double], userAccY: [Double], userAccZ: [Double],
                         rawAccX: [Double], rawAccY: [Double], rawAccZ: [Double],
                         gyroX: [Double], gyroY: [Double], gyroZ: [Double]) -> MLMultiArray? {
        
        guard userAccX.count == windowSize else { return nil }
        
        // 🟢 حالت مستقیم: بدون هیچ‌گونه جابه‌جایی یا چرخش پنهان
        let ba_x = userAccX
        let ba_y = userAccY
        let ba_z = userAccZ

        let tot_x = rawAccX
        let tot_y = rawAccY
        let tot_z = rawAccZ

        let gy_x = gyroX
        let gy_y = gyroY
        let gy_z = gyroZ

        do {
            let featureArray = try MLMultiArray(shape: [1, NSNumber(value: featureCount)], dataType: .double)
            var allFeatures: [Double] = []
            
            let channels = [tot_x, tot_y, tot_z, ba_x, ba_y, ba_z, gy_x, gy_y, gy_z]
            
            for signal in channels {
                let channelFeats = computePython8Features(signal)
                allFeatures.append(contentsOf: channelFeats)
            }
            
            let sma_total = calculateSMA(x: tot_x, y: tot_y, z: tot_z)
            let sma_body = calculateSMA(x: ba_x, y: ba_y, z: ba_z)
            let sma_gyro = calculateSMA(x: gy_x, y: gy_y, z: gy_z)
            
            allFeatures.append(contentsOf: [sma_total, sma_body, sma_gyro])
            
            for i in 0..<featureCount {
                featureArray[i] = NSNumber(value: allFeatures[i])
            }
            
            return featureArray
        } catch {
            return nil
        }
    }

    private func computePython8Features(_ sig: [Double]) -> [Double] {
        let n = vDSP_Length(windowSize)
        let nDouble = Double(windowSize)
        
        var mean = 0.0; vDSP_meanvD(sig, 1, &mean, n)
        var ms = 0.0; vDSP_measqvD(sig, 1, &ms, n)
        let std = sqrt(max(0, ms - (mean * mean)))
        var mx = 0.0; vDSP_maxvD(sig, 1, &mx, n)
        
        var skew = 0.0
        var kurt = -3.0
        
        if std > 1e-8 {
            var sum3 = 0.0
            var sum4 = 0.0
            for x in sig {
                let diff = (x - mean) / std
                let d2 = diff * diff
                sum3 += d2 * diff
                sum4 += d2 * d2
            }
            skew = sum3 / nDouble
            kurt = (sum4 / nDouble) - 3.0
        }
        
        // 🟢 استخراج دقیق FFT بدون باگ vDSP
        let sigFFT = computeExactPythonFFT(signal: sig)
        let nF = vDSP_Length(sigFFT.count)
        
        var fftMean = 0.0; vDSP_meanvD(sigFFT, 1, &fftMean, nF)
        var fftMs = 0.0; vDSP_measqvD(sigFFT, 1, &fftMs, nF)
        let fftStd = sqrt(max(0, fftMs - (fftMean * fftMean)))
        
        let psd = sigFFT.map { ($0 * $0) / 64.0 }
        let sumPSD = psd.reduce(0, +)
        
        var entropy = 0.0
        for p in psd {
            let pNorm = p / (sumPSD + 1e-12)
            entropy += pNorm * log2(pNorm + 1e-12)
        }
        entropy = -entropy
        
        return [mean, std, mx, skew, kurt, fftMean, fftStd, entropy]
    }

    // 🟢 این تابع جایگزین شد تا دقیقاً خروجی np.abs(fft(sig))[:64] را تولید کند
    // 🟢 جایگزین تابع قبلی برای رفع خطای Pointer در سوئیفت
        private func computeExactPythonFFT(signal: [Double]) -> [Double] {
            var real = signal
            var imag = [Double](repeating: 0.0, count: windowSize)
            var magnitudes = [Double](repeating: 0.0, count: 64)
            
            // ایجاد بلوک ایمن حافظه برای آرایه‌ها
            real.withUnsafeMutableBufferPointer { realPtr in
                imag.withUnsafeMutableBufferPointer { imagPtr in
                    guard let realBase = realPtr.baseAddress, let imagBase = imagPtr.baseAddress else { return }
                    
                    var split = DSPDoubleSplitComplex(realp: realBase, imagp: imagBase)
                    
                    vDSP_fft_zipD(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                    
                    // DC Component (اندیس 0)
                    magnitudes[0] = abs(split.realp[0])
                    
                    // فرکانس‌های 1 تا 63
                    for i in 1..<64 {
                        let r = split.realp[i]
                        let im = split.imagp[i]
                        magnitudes[i] = sqrt(r * r + im * im)
                    }
                }
            }
            
            return magnitudes
        }

    private func calculateSMA(x: [Double], y: [Double], z: [Double]) -> Double {
        var sum = 0.0
        for i in 0..<x.count { sum += abs(x[i]) + abs(y[i]) + abs(z[i]) }
        return sum / Double(x.count)
    }
}
