import Foundation

struct CSVManagerUrl {
    
    static func loadDataAndRunInference(fileURL: URL) -> [ProcessedSegment]? {
        do {
            let fileContents = try String(contentsOf: fileURL)
            print("Success: File loaded from temporary storage. Parsing data...")
            
            var parsedData = [[Double]]()
            let rows = fileContents.components(separatedBy: .newlines)
            
            for row in rows {
                guard !row.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
                
                let columns = row.components(separatedBy: ",")
                let doubleValues = columns.map { Double($0.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0.0 }
                
                if doubleValues.count >= 13 {
                    parsedData.append(doubleValues)
                }
            }
            
            // --- NOISE REDUCTION LOGIC (Handling Noise Cropping) ---
            // 3 seconds * 50 Hz = 150 samples to remove from start and end
            let trimCount = 0
            var finalDataToProcess = parsedData
            
            // Ensure the recording is long enough to be trimmed (longer than 6 seconds)
            if parsedData.count > (trimCount * 2) {
                finalDataToProcess = Array(parsedData[trimCount..<(parsedData.count - trimCount)])
                print("Trimmed \(trimCount) rows from start and end. Original size: \(parsedData.count), New size: \(finalDataToProcess.count)")
            } else {
                print("Warning: Recording is too short (under 6 seconds). Processing without trimming.")
            }
            // -------------------------------------------------------
            
            print("Successfully parsed \(finalDataToProcess.count) rows. Starting Batch Inference...")
            
            let verifier = NeuralInferenceVerifierUrl()
            return verifier.runBatchInference(fromCSV: finalDataToProcess)
            
        } catch {
            print("Error: Could not read the file contents.")
            return nil
        }
    }
}
