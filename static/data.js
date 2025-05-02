const trainButton = document.getElementById('train-button');
const predictButton = document.getElementById('predict-button');
const sqftInput = document.getElementById('sqft');
const statusSpan = document.getElementById('status');
const predictStatusSpan = document.getElementById('predict-status');
const predictionResultSpan = document.getElementById('prediction-result');
const ctx = document.getElementById('regressionChart').getContext('2d');
const plotPlaceholder = document.getElementById('plot-placeholder');
let regressionChart;

// Function to initialize or update the chart
function updateChart(originalData, fittedLineData) {
    const chartData = {
        datasets: [
            {
                label: 'Original Data',
                data: originalData, // Expects array of {x, y} objects
                backgroundColor: 'rgba(0, 123, 255, 0.5)',
                type: 'scatter', // Specify type for scatter
                pointRadius: 3,
            },
            {
                label: 'Fitted Line',
                data: fittedLineData, // Expects array of {x, y} objects
                borderColor: 'rgba(255, 0, 0, 1)',
                borderWidth: 2,
                fill: false,
                type: 'line', // Specify type for line
                pointRadius: 0, // Hide points on the line
                tension: 0.1 // Optional: slight curve
            }
        ]
    };

    const config = {
        // type: 'scatter', // Default type, can be overridden by datasets
        data: chartData,
        options: {
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'Square Feet Living'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Price'
                    },
                    ticks: { // Format Y-axis ticks as currency
                        callback: function (value, index, values) {
                            return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value);
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: { // Format tooltip
                        label: function (context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(context.parsed.y);
                            }
                            label += ` (${context.parsed.x.toLocaleString()} sqft)`;
                            return label;
                        }
                    }
                }
            }
        }
    };

    if (regressionChart) {
        // If chart exists, update its data and redraw
        regressionChart.data = chartData;
        regressionChart.update();
    } else {
        // Otherwise, create a new chart instance
        regressionChart = new Chart(ctx, config);
    }
    plotPlaceholder.style.display = 'none'; // Hide placeholder if shown
    document.getElementById('regressionChart').style.display = 'block'; // Ensure canvas is visible
}

// Initialize chart placeholder state
document.getElementById('regressionChart').style.display = 'none';
plotPlaceholder.style.display = 'block';


trainButton.addEventListener('click', async () => {
    statusSpan.textContent = 'Initializing training...'; // Initial message
    statusSpan.className = '';
    trainButton.disabled = true;
    predictButton.disabled = true;
    predictStatusSpan.textContent = 'Training in progress...';
    predictStatusSpan.className = '';
    predictionResultSpan.textContent = '---'; // Reset prediction

    let finalResult = null; // To store the final result message

    try {
        const response = await fetch('/train', { method: 'POST' });

        if (!response.ok) {
            // Handle HTTP errors (e.g., 500 Internal Server Error, 404 Not Found)
            let errorMsg = `Server error: ${response.status} ${response.statusText}`;
            try {
                // Try to parse a JSON error message from the server if available
                const errData = await response.json();
                // Check if the error response follows the {type: 'result', ...} structure
                if (errData && errData.type === 'result' && errData.message) {
                    errorMsg = `Error: ${errData.message}`;
                } else {
                    errorMsg = `Error: ${errData.message || errorMsg}`;
                }
            } catch (e) {
                // Ignore if response body is not JSON or empty
            }
            throw new Error(errorMsg); // Throw an error to be caught by the catch block
        }

        // Process the streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) {
                // Process any remaining data in the buffer when the stream ends
                if (buffer.trim()) {
                    try {
                        const data = JSON.parse(buffer);
                        if (data.type === 'progress') {
                            if (data.message) {
                                statusSpan.textContent = data.message;
                            } else {
                                statusSpan.textContent = `Epoch ${data.epoch}/${data.epochs}, Loss: ${data.loss}`;
                            }
                        } else if (data.type === 'result') {
                            finalResult = data; // Store the final result
                        }
                    } catch (e) {
                        console.error('Error parsing final chunk:', e, buffer);
                        // Decide how to handle parsing error for the last bit
                        finalResult = { success: false, message: "Error processing final training update." };
                    }
                }
                break; // Exit the loop
            }

            // Append new data chunk to buffer and decode
            buffer += decoder.decode(value, { stream: true });

            // Process complete lines (newline-delimited JSON)
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep the last (potentially incomplete) line in the buffer

            lines.forEach(line => {
                if (line.trim() === '') return; // Skip empty lines
                try {
                    const data = JSON.parse(line);
                    // Check the type of message
                    if (data.type === 'progress') {
                        if (data.message) {
                            statusSpan.textContent = data.message;
                        } else {
                            // Update status with epoch and loss
                            statusSpan.textContent = `Epoch ${data.epoch}/${data.epochs}, Loss: ${data.loss}`;
                        }
                    } else if (data.type === 'result') {
                        finalResult = data; // Store the final result message
                        // Don't break here, let the stream finish naturally
                    }
                } catch (e) {
                    console.error('Error parsing JSON line:', e, line);
                    // Optionally update statusSpan with a parsing error message
                    // statusSpan.textContent = 'Error processing training update.';
                    // statusSpan.className = 'error';
                }
            });
        } // End while loop

        // After stream has finished, process the final result
        if (finalResult) {
            if (finalResult.success) {
                statusSpan.textContent = 'Training complete!';
                statusSpan.className = 'success';
                updateChart(finalResult.original_data, finalResult.fitted_line);
                predictButton.disabled = false;
                predictStatusSpan.textContent = 'Model ready for predictions.';
                predictStatusSpan.className = 'success';
            } else {
                statusSpan.textContent = `Error: ${finalResult.message || 'Training failed.'}`;
                statusSpan.className = 'error';
                predictStatusSpan.textContent = 'Training failed. Cannot predict.';
                predictStatusSpan.className = 'error';
            }
        } else {
            // This case might happen if the stream ended unexpectedly without a 'result' message
            throw new Error('Training finished without a final result.');
        }

    } catch (error) {
        console.error('Training Fetch/Stream error:', error);
        statusSpan.textContent = `Error: ${error.message || 'Could not connect or process training.'}`;
        statusSpan.className = 'error';
        predictStatusSpan.textContent = 'Training error. Cannot predict.';
        predictStatusSpan.className = 'error';
        // Ensure predict button remains disabled if training failed critically
        predictButton.disabled = true;
    } finally {
        // Re-enable train button regardless of outcome
        trainButton.disabled = false;
        // Predict button state is handled within the try/catch based on finalResult.success
    }
});

// --- Keep predictButton event listener as is ---
predictButton.addEventListener('click', async () => {
    const sqftValue = sqftInput.value;
    if (!sqftValue) {
        predictStatusSpan.textContent = 'Please enter square footage.';
        predictStatusSpan.className = 'error';
        return;
    }

    predictStatusSpan.textContent = 'Predicting...';
    predictStatusSpan.className = '';
    predictionResultSpan.textContent = '---'; // Reset prediction
    predictButton.disabled = true; // Disable while predicting

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sqft: sqftValue })
        });
        const result = await response.json();

        if (response.ok && result.success) {
            predictionResultSpan.textContent = result.prediction;
            predictStatusSpan.textContent = 'Prediction successful.';
            predictStatusSpan.className = 'success';
        } else {
            predictStatusSpan.textContent = `Error: ${result.message || 'Prediction failed.'}`;
            predictStatusSpan.className = 'error';
        }
    } catch (error) {
        console.error('Fetch error:', error);
        predictStatusSpan.textContent = 'Error: Could not connect to server.';
        predictStatusSpan.className = 'error';
    } finally {
        predictButton.disabled = false;
    }
});

// --- Keep sqftInput event listener as is ---
sqftInput.addEventListener('input', () => {
    if (predictStatusSpan.className === 'error' && predictStatusSpan.textContent === 'Please enter square footage.') {
        predictStatusSpan.textContent = '';
        predictStatusSpan.className = '';
    }
});