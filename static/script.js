let currentDataset = "correlation"; // Default dataset

// Function to update the dataset and fetch a new graph
function showGraphs(dataset) {
    currentDataset = dataset; // Update the global dataset variable
    fetchGraph(); // Call the function to fetch the graph
}

// Function to fetch the graph when a dataset or graph type is changed
function fetchGraph() {
    let graphType = document.getElementById("graphType").value;

    fetch("/get_graph", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ graphType: graphType, dataset: currentDataset })
    })
    .then(response => response.json())
    .then(data => {
        let graphContainer = document.getElementById("graph-container");
        if (data.image) {
            graphContainer.innerHTML = `<img src="data:image/png;base64,${data.image}" style="max-width: 80%;" />`;
        } else {
            graphContainer.innerHTML = `<p>Error loading graph</p>`;
        }
    })
    .catch(error => console.error("Error fetching graph:", error));
}

// Event listener to change graph type dynamically
document.getElementById("graphType").addEventListener("change", fetchGraph);
