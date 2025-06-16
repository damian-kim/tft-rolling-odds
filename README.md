# TFT Rolling Odds Calculator

A web-based calculator for Teamfight Tactics (TFT) that helps players calculate the probability of finding specific units during a rolldown.

## Features

- Calculate probabilities for finding specific units based on:
  - Current level
  - Gold to roll
  - Unit cost
  - Units already taken from the pool
- Interactive sliders for easy input
- Visual representation with bar charts
- Rolldown analysis for multiple units
- Mobile-friendly responsive design

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask app:
```bash
python app.py
```

3. Open your browser to `http://localhost:5000`

## Deployment

The application is configured for deployment to GitHub Pages. To deploy:

1. Create a new GitHub repository
2. Push your code to the repository:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

3. Enable GitHub Pages in your repository settings:
   - Go to Settings > Pages
   - Select the `gh-pages` branch as the source
   - Save the settings

4. The GitHub Action will automatically build and deploy your site when you push to the main branch.

## Contributing

Feel free to submit issues and enhancement requests! 