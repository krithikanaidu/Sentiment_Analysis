let lastResult = null;
let delayTimeout = null;

document.getElementById('reviewForm').addEventListener('submit', async (event) => {
    event.preventDefault();

    const review = document.getElementById('review').value;

    if (!review) {
        document.getElementById('result').textContent = 'Please enter a review.';
        document.getElementById('result').classList.remove('repeated');
        lastResult = null; 
        return;
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ review }),
        });

        const data = await response.json();

        if (response.ok) {
            const currentResult = `Sentiment: ${data.sentiment}`;
            const resultElement = document.getElementById('result');

            
            if (delayTimeout) {
                clearTimeout(delayTimeout);
                delayTimeout = null;
            }

            if (currentResult === lastResult) {
               
                resultElement.textContent = ''; 
                delayTimeout = setTimeout(() => {
                    resultElement.textContent = currentResult;
                    resultElement.classList.add('repeated');
                }, 50); 
            } else {
                
                resultElement.textContent = currentResult;
                resultElement.classList.remove('repeated');
            }

            lastResult = currentResult;
        } else {
            document.getElementById('result').textContent = data.error || 'Error analyzing sentiment.';
            document.getElementById('result').classList.remove('repeated');
            lastResult = null;
        }
    } catch (error) {
        document.getElementById('result').textContent = 'An error occurred. Please try again.';
        document.getElementById('result').classList.remove('repeated');
        lastResult = null;
    }
});
