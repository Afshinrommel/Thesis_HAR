import Foundation
import Accelerate

class FeatureExtractorf {
    let windowSize = 128
    let numChannels = 9
    private let log2n = vDSP_Length(7)
    private let fftSetup: FFTSetupD

    init() {
        guard let setup = vDSP_create_fftsetupD(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("❌ FFT Setup Failed")
        }
        self.fftSetup = setup
    }

    deinit { vDSP_destroy_fftsetupD(fftSetup) }

    // MARK: - تابع جدید استخراج ۷۵ ویژگی (RF UCI)
    func extract75FeaturesUCI(window: [[Double]]) -> [Double] {
        var allFeatures = [Double]()
        
        // ۱. تحلیل ۹ کانال (۹ کانال × ۸ ویژگی = ۷۲ ویژگی)
        for c in 0..<numChannels {
            let sig = window[c]
            let n = vDSP_Length(windowSize)

            // الف) ویژگی‌های زمانی (۵ مورد)
            var m = 0.0; vDSP_meanvD(sig, 1, &m, n)
            var ms = 0.0; vDSP_measqvD(sig, 1, &ms, n)
            let std = sqrt(max(0, ms - (m*m)))
            var mx = 0.0; vDSP_maxvD(sig, 1, &mx, n)
            
            let skew = calculateSkewness(sig, mean: m, std: std)
            let kurt = calculateKurtosis(sig, mean: m, std: std)
            
            allFeatures.append(contentsOf: [m, std, mx, skew, kurt])

            // ب) ویژگی‌های فرکانسی (۳ مورد)
            let mags = computeFFT(signal: sig)
            let sigFFT = Array(mags[0...63]) // مطابق پایتون [:64]
            let nF = vDSP_Length(sigFFT.count)
            
            var meanF = 0.0; vDSP_meanvD(sigFFT, 1, &meanF, nF)
            var msF = 0.0; vDSP_measqvD(sigFFT, 1, &msF, nF)
            let stdF = sqrt(max(0, msF - (meanF * meanF)))
            
            // محاسبه آنتروپی طیفی (Spectral Entropy)
            let psd = sigFFT.map { ($0 * $0) / 64.0 }
            let sumPSD = psd.reduce(0, +)
            let psdNorm = psd.map { $0 / (sumPSD + 1e-12) }
            let entropy = -psdNorm.reduce(0) { $0 + ($1 * log2($1 + 1e-12)) }
            
            allFeatures.append(contentsOf: [meanF, stdF, entropy])
        }
        
        // ۲. ویژگی‌های جهانی SMA (۳ مورد)
        // SMA Total (کانال 0 تا 2)، Body (3 تا 5)، Gyro (6 تا 8)
        let smaTotal = calculateSMA(channels: Array(window[0...2]))
        let smaBody = calculateSMA(channels: Array(window[3...5]))
        let smaGyro = calculateSMA(channels: Array(window[6...8]))
        
        allFeatures.append(contentsOf: [smaTotal, smaBody, smaGyro])

        return allFeatures // مجموعاً ۷۵ ویژگی
    }

    // MARK: - Helper Functions
    
    private func calculateSkewness(_ data: [Double], mean: Double, std: Double) -> Double {
        guard std > 1e-6 else { return 0.0 }
        let sumCube = data.reduce(0.0) { $0 + pow($1 - mean, 3) }
        return (sumCube / Double(data.count)) / pow(std, 3)
    }

    private func calculateKurtosis(_ data: [Double], mean: Double, std: Double) -> Double {
        guard std > 1e-6 else { return 0.0 }
        let sumQuart = data.reduce(0.0) { $0 + pow($1 - mean, 4) }
        return ((sumQuart / Double(data.count)) / pow(std, 4)) - 3.0 // Fisher definition
    }

    private func calculateSMA(channels: [[Double]]) -> Double {
        var totalAbsSum = 0.0
        for t in 0..<windowSize {
            var sumAtT = 0.0
            for c in 0..<channels.count {
                sumAtT += abs(channels[c][t])
            }
            totalAbsSum += sumAtT
        }
        return totalAbsSum / Double(windowSize)
    }

    private func computeFFT(signal: [Double]) -> [Double] {
        var real = signal, imag = [Double](repeating: 0.0, count: windowSize)
        var split = DSPDoubleSplitComplex(realp: &real, imagp: &imag)
        vDSP_fft_zipD(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
        var mags = [Double](repeating: 0.0, count: windowSize/2)
        vDSP_zvabsD(&split, 1, &mags, 1, vDSP_Length(windowSize/2))
        return mags
    }
}
