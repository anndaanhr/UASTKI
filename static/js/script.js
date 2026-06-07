document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const searchInput = document.getElementById("searchInput");
    const searchBtn = document.getElementById("searchBtn");
    const resultsGrid = document.getElementById("resultsGrid");
    const resultsCount = document.getElementById("resultsCount");
    const emptyState = document.getElementById("emptyState");
    const loadingState = document.getElementById("loadingState");
    const refreshEvalBtn = document.getElementById("refreshEvalBtn");
    const bentoGrid = document.querySelector(".bento-grid");

    // Load Evaluation Metrics on mount
    fetchEvaluationMetrics();

    // Event Listeners for Search
    searchBtn.addEventListener("click", () => performSearch(searchInput.value));
    searchInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") performSearch(searchInput.value);
    });

    // Refresh Evaluation
    refreshEvalBtn.addEventListener("click", () => {
        refreshEvalBtn.style.animation = "spin 1s linear";
        fetchEvaluationMetrics().then(() => {
            setTimeout(() => { refreshEvalBtn.style.animation = "none"; }, 500);
        });
    });

    // --- API Calls ---

    async function fetchEvaluationMetrics() {
        try {
            const response = await fetch('/api/evaluate');
            const data = await response.json();
            renderBentoGrid(data.summary);
        } catch (error) {
            console.error("Failed to load evaluation metrics:", error);
            bentoGrid.innerHTML = `<div class="bento-item"><p style="color:red">Failed to load metrics.</p></div>`;
        }
    }

    const queryAnalysis = document.getElementById("queryAnalysis");

    async function performSearch(query) {
        if (!query.trim()) return;

        // UI State: Loading
        resultsGrid.innerHTML = "";
        queryAnalysis.innerHTML = "";
        queryAnalysis.classList.add("hidden");
        emptyState.classList.add("hidden");
        loadingState.classList.remove("hidden");
        resultsCount.textContent = `Searching for "${query}"...`;

        try {
            const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
            const data = await response.json();
            
            loadingState.classList.add("hidden");
            
            if (data.results && data.results.length > 0) {
                resultsCount.textContent = `Found ${data.results.length} results`;
                
                // Show query analysis
                queryAnalysis.classList.remove("hidden");
                const tokenHTML = data.query_tokens.map(t => `<span class="token-badge">${t}</span>`).join("");
                queryAnalysis.innerHTML = `
                    <div class="analysis-row">
                        <span class="analysis-label">Parsed Tokens:</span>
                        <div class="token-container">${tokenHTML || 'None'}</div>
                    </div>
                    <div class="analysis-row time-row">
                        <span class="analysis-label"><i class="ph ph-clock"></i> Executed in:</span>
                        <span>${data.execution_time_ms} ms</span>
                    </div>
                `;

                renderResults(data.results, data.query_tokens);
            } else {
                resultsCount.textContent = "";
                emptyState.classList.remove("hidden");
            }
        } catch (error) {
            console.error("Search failed:", error);
            loadingState.classList.add("hidden");
            resultsCount.textContent = "An error occurred while searching.";
        }
    }

    // --- Render Functions ---

    function renderBentoGrid(summary) {
        // Formatting function
        const formatScore = (val) => (val * 100).toFixed(1) + "%";

        const metrics = [
            { label: "MAP", value: formatScore(summary.MAP), desc: "Mean Average Precision over all queries" },
            { label: "P@5", value: formatScore(summary['Avg_Precision@5']), desc: "Average Precision at top 5 results" },
            { label: "P@10", value: formatScore(summary['Avg_Precision@10']), desc: "Average Precision at top 10 results" },
            { label: "NDCG", value: formatScore(summary['Avg_NDCG@10']), desc: "Normalized Discounted Cumulative Gain" }
        ];

        bentoGrid.innerHTML = metrics.map(m => `
            <div class="bento-item">
                <div class="bento-label">${m.label}</div>
                <div class="bento-value">${m.value}</div>
                <div class="bento-desc">${m.desc}</div>
            </div>
        `).join('');
    }

    // --- Terminal & Modal ---
    const terminalBody = document.getElementById("terminalBody");
    const toggleTerminalBtn = document.getElementById("toggleTerminalBtn");
    const terminalContainer = document.getElementById("terminalContainer");
    
    toggleTerminalBtn.addEventListener("click", () => {
        terminalContainer.classList.toggle("collapsed");
        toggleTerminalBtn.innerHTML = terminalContainer.classList.contains("collapsed") ? 
            '<i class="ph ph-caret-up"></i>' : '<i class="ph ph-caret-down"></i>';
    });

    function logToTerminal(message, type = "system") {
        const line = document.createElement("div");
        line.className = `log-line ${type}`;
        line.textContent = message;
        terminalBody.appendChild(line);
        terminalBody.scrollTop = terminalBody.scrollHeight;
    }

    const movieModal = document.getElementById("movieModal");
    const closeModalBtn = document.getElementById("closeModalBtn");
    
    function closeMyModal() {
        movieModal.classList.remove("active");
        setTimeout(() => movieModal.classList.add("hidden"), 300);
    }

    closeModalBtn.addEventListener("click", closeMyModal);
    
    // Close modal on outside click
    movieModal.addEventListener("click", (e) => {
        if(e.target === movieModal) closeMyModal();
    });

    function openModal(movie, tokens) {
        document.getElementById("modalTitle").innerHTML = highlightText(movie.title, tokens);
        document.getElementById("modalYear").textContent = movie.year || "Unknown Year";
        document.getElementById("modalRating").textContent = movie.rating || "Unrated";
        document.getElementById("modalScore").innerHTML = `<i class="ph-fill ph-target"></i> Score: ${movie.relevance_score.toFixed(3)}`;
        
        // Clean up array string if it exists
        let directorsStr = movie.directors;
        let actorsStr = movie.actors;
        
        if (typeof directorsStr === 'string' && directorsStr.startsWith('[')) {
            try { directorsStr = JSON.parse(directorsStr.replace(/'/g, '"')).join(', '); } catch(e){}
        }
        if (typeof actorsStr === 'string' && actorsStr.startsWith('[')) {
            try { actorsStr = JSON.parse(actorsStr.replace(/'/g, '"')).join(', '); } catch(e){}
        }

        document.getElementById("modalDirectors").textContent = directorsStr || "Unknown";
        document.getElementById("modalActors").textContent = actorsStr || "Unknown";
        document.getElementById("modalDesc").innerHTML = highlightText(movie.description, tokens);
        
        movieModal.classList.remove("hidden");
        // small delay to allow display:block to apply before animating opacity
        setTimeout(() => movieModal.classList.add("active"), 10);
    }

    let currentResults = [];
    let currentTokens = [];

    // Highlight search terms in the text
    function highlightText(text, tokens) {
        if (!tokens || tokens.length === 0) return text;
        let highlighted = text;
        
        // simple case insensitive replace
        tokens.forEach(token => {
            if (token.length > 2) { // Only highlight words > 2 chars
                const regex = new RegExp(`(${token})`, 'gi');
                highlighted = highlighted.replace(regex, '<span class="text-highlight">$1</span>');
            }
        });
        return highlighted;
    }

    function renderResults(movies, tokens) {
        currentResults = movies;
        currentTokens = tokens;
        resultsGrid.innerHTML = "";
        
        movies.forEach((movie, index) => {
            const card = document.createElement("div");
            card.className = "movie-card animate-fade-up";
            card.style.animationDelay = `${index * 0.05}s`;
            
            card.innerHTML = `
                <div class="movie-title">${highlightText(movie.title, tokens)}</div>
                <div class="movie-meta">
                    <span class="movie-year">${movie.year || 'Unknown'}</span>
                    <span class="movie-genre">${movie.genre.split(',')[0] || 'Movie'}</span>
                    <span class="movie-score">
                        <i class="ph-fill ph-target"></i> 
                        ${movie.relevance_score.toFixed(2)}
                    </span>
                </div>
                <div class="movie-desc">${highlightText(movie.description, tokens)}</div>
            `;
            
            card.addEventListener("click", () => openModal(movie, tokens));
            resultsGrid.appendChild(card);
        });
    }

    // Override performSearch to add logs
    const originalPerformSearch = performSearch;
    performSearch = async function(query) {
        logToTerminal(`[USER] Search initiated for: "${query}"`, "highlight");
        await originalPerformSearch(query);
        logToTerminal(`[SYS] Rendered UI completely. Ready.`, "system");
    }
});
