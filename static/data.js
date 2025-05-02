const trainButton = document.getElementById('train-button');
const predictButton = document.getElementById('predict-button');
const sqftInput = document.getElementById('sqft');
const statusSpan = document.getElementById('status');
const predictStatusSpan = document.getElementById('predict-status');
const predictionResultSpan = document.getElementById('prediction-result');
// Get canvas context
const ctx = document.getElementById('regressionChart').getContext('2d');
const plotPlaceholder = document.getElementById('plot-placeholder');
let regressionChart; // Variable to hold the chart instance

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
    statusSpan.textContent = 'Training... please wait.';
    statusSpan.className = '';
    trainButton.disabled = true;
    predictButton.disabled = true;

    try {
        const response = await fetch('/train', { method: 'POST' });
        const result = await response.json(); // Expect JSON with plot data

        if (response.ok && result.success) {
            statusSpan.textContent = 'Training complete!';
            statusSpan.className = 'success';
            // Update the chart with received data
            updateChart(result.original_data, result.fitted_line);
            predictButton.disabled = false;
            predictStatusSpan.textContent = 'Model ready for predictions.';
            predictStatusSpan.className = 'success';
        } else {
            statusSpan.textContent = `Error: ${result.message || 'Training failed.'}`;
            statusSpan.className = 'error';
            predictStatusSpan.textContent = 'Training failed. Cannot predict.';
            predictStatusSpan.className = 'error';
        }
    } catch (error) {
        console.error('Fetch error:', error);
        statusSpan.textContent = 'Error: Could not connect to server.';
        statusSpan.className = 'error';
        predictStatusSpan.textContent = 'Server error. Cannot predict.';
        predictStatusSpan.className = 'error';
    } finally {
        trainButton.disabled = false;
        if (!predictButton.disabled && statusSpan.className !== 'success') {
            predictButton.disabled = true;
        }
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