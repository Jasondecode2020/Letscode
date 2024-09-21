// Example ratings array and problem names
let problems = [
  { name: "Problem 1", rating: 1200 },
  { name: "Problem 2", rating: 1500 },
  { name: "Problem 3", rating: 1800 },
  { name: "Problem 4", rating: 2000 },
];

// Initialize chart
const ctx = document.getElementById('ratingChart').getContext('2d');
let ratingChart = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: problems.map(p => p.name), // Extract problem names for labels
    datasets: [{
      label: 'LeetCode Problem Rating',
      data: problems.map(p => p.rating), // Extract ratings for data
      backgroundColor: 'rgba(75, 192, 192, 0.2)',
      borderColor: 'rgba(75, 192, 192, 1)',
      borderWidth: 1
    }]
  },
  options: {
    scales: {
      y: {
        beginAtZero: true
      }
    }
  }
});

// Function to dynamically update chart with new ratings
function updateChart(newProblems) {
  ratingChart.data.labels = newProblems.map(p => p.name); // Update labels
  ratingChart.data.datasets[0].data = newProblems.map(p => p.rating); // Update data
  ratingChart.update(); // Redraw the chart
}

// Simulate a dynamic update (You can replace this with actual data updates)
setTimeout(() => {
  let newProblems = [
    { name: "Problem 1", rating: 1300 },
    { name: "Problem 2", rating: 1600 },
    { name: "Problem 3", rating: 1900 },
    { name: "Problem 4", rating: 2100 },
  ];
  updateChart(newProblems);
}, 5000); // Updates the chart after 5 seconds