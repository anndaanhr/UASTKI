document.addEventListener("DOMContentLoaded", () => {
    // --- DOM Elements ---
    
    // Home State
    const homeSearchArea = document.getElementById("homeSearchArea");
    const searchInputHome = document.getElementById("searchInputHome");
    const searchBtnHome = document.getElementById("searchBtnHome");
    const clearBtnHome = document.getElementById("clearBtnHome");

    // Results State (Top Header)
    const topHeader = document.getElementById("topHeader");
    const searchInputTop = document.getElementById("searchInputTop");
    const searchBtnTop = document.getElementById("searchBtnTop");
    const clearBtnTop = document.getElementById("clearBtnTop");
    
    // Results Area
    const resultsArea = document.getElementById("resultsArea");
    const resultsStats = document.getElementById("resultsStats");
    const aiOverview = document.getElementById("aiOverview");
    const aiSummaryText = document.getElementById("aiSummaryText");
    const loadingState = document.getElementById("loadingState");
    const emptyState = document.getElementById("emptyState");
    const resultsList = document.getElementById("resultsList");

    // Knowledge Panel
    const knowledgePanel = document.getElementById("knowledgePanel");
    const closePanelBtn = document.getElementById("closePanelBtn");
    
    // --- State Variables ---
    let isHome = true;

    // --- Event Listeners ---

    // Home Input Events
    searchInputHome.addEventListener("input", () => {
        clearBtnHome.classList.toggle("hidden", searchInputHome.value.length === 0);
    });
    clearBtnHome.addEventListener("click", () => {
        searchInputHome.value = "";
        clearBtnHome.classList.add("hidden");
        searchInputHome.focus();
    });
    searchInputHome.addEventListener("keypress", (e) => {
        if (e.key === "Enter") performSearch(searchInputHome.value);
    });
    searchBtnHome.addEventListener("click", () => performSearch(searchInputHome.value));

    // Top Input Events
    searchInputTop.addEventListener("input", () => {
        clearBtnTop.classList.toggle("hidden", searchInputTop.value.length === 0);
    });
    clearBtnTop.addEventListener("click", () => {
        searchInputTop.value = "";
        clearBtnTop.classList.add("hidden");
        searchInputTop.focus();
    });
    searchInputTop.addEventListener("keypress", (e) => {
        if (e.key === "Enter") performSearch(searchInputTop.value);
    });
    searchBtnTop.addEventListener("click", () => performSearch(searchInputTop.value));

    // Panel Event
    closePanelBtn.addEventListener("click", () => {
        knowledgePanel.classList.add("hidden");
    });

    // --- Search Logic ---

    async function performSearch(query) {
        if (!query.trim()) return;

        // Transition from Home to Results view
        if (isHome) {
            homeSearchArea.classList.add("hidden");
            document.body.classList.remove("home-state");
            topHeader.classList.remove("hidden");
            resultsArea.classList.remove("hidden");
            isHome = false;
        }

        // Sync inputs
        searchInputTop.value = query;
        clearBtnTop.classList.remove("hidden");

        // UI Loading State
        resultsList.innerHTML = "";
        emptyState.classList.add("hidden");
        aiOverview.classList.add("hidden");
        knowledgePanel.classList.add("hidden");
        resultsStats.textContent = "";
        loadingState.classList.remove("hidden");

        try {
            const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
            const data = await response.json();
            
            loadingState.classList.add("hidden");
            
            if (data.results && data.results.length > 0) {
                // Render Stats
                const seconds = (data.execution_time_ms / 1000).toFixed(2);
                resultsStats.textContent = `About ${data.results.length} results (${seconds} seconds)`;
                
                // Render AI Overview
                generateAIOverview(query, data.query_tokens, data.results);
                
                // Render Results
                renderResults(data.results, data.query_tokens);
            } else {
                emptyState.classList.remove("hidden");
            }
        } catch (error) {
            console.error("Search failed:", error);
            loadingState.classList.add("hidden");
            resultsStats.textContent = "An error occurred while searching.";
        }
    }

    // --- Render Functions ---

    function generateAIOverview(query, tokens, results) {
        const topSources = [...new Set(results.slice(0, 3).map(r => r.source || 'academic journals'))];
        const sourceStr = topSources.join(", ");
        
        let aiText = `Based on your search for <b>"${query}"</b>, we found extensive literature matching key concepts like <i>${tokens.join(", ")}</i>. `;
        aiText += `The top resulting research papers are primarily published in ${sourceStr}. `;
        aiText += `These documents discuss various methodologies and findings related to your query.`;
        
        aiSummaryText.innerHTML = aiText;
        aiOverview.classList.remove("hidden");
    }

    function highlightText(text, tokens) {
        if (!tokens || tokens.length === 0) return text;
        let highlighted = text;
        
        // simple case insensitive replace
        tokens.forEach(token => {
            if (token.length > 2) { 
                const regex = new RegExp(`(${token})`, 'gi');
                highlighted = highlighted.replace(regex, '<b>$1</b>');
            }
        });
        return highlighted;
    }

    function renderResults(papers, tokens) {
        resultsList.innerHTML = "";
        
        papers.forEach(paper => {
            const item = document.createElement("div");
            item.className = "result-item";
            
            // Format Author
            let authorStr = paper.author;
            if (typeof authorStr === 'string' && authorStr.startsWith('[')) {
                try { authorStr = JSON.parse(authorStr.replace(/'/g, '"')).join(', '); } catch(e){}
            }

            // Create URL display
            let displayUrl = paper.url || "https://researcharchive.org/...";
            if(displayUrl.length > 40) displayUrl = displayUrl.substring(0, 40) + "...";

            item.innerHTML = `
                <a href="javascript:void(0)" class="result-source">
                    <div class="source-icon">${(paper.source || 'R')[0].toUpperCase()}</div>
                    <div class="source-info">
                        <div class="source-text">${paper.source || 'Research Paper'}</div>
                        <div class="source-url">${displayUrl}</div>
                    </div>
                </a>
                <a href="javascript:void(0)" class="result-title">${highlightText(paper.title, tokens)}</a>
                <div class="result-snippet">
                    <span class="meta-date">${paper.date || '2025'} —</span>
                    ${highlightText(paper.text, tokens)}
                </div>
            `;
            
            item.querySelector('.result-title').addEventListener("click", () => openKnowledgePanel(paper, authorStr));
            
            resultsList.appendChild(item);
        });
    }

    function openKnowledgePanel(paper, authorStr) {
        document.getElementById("panelTitle").textContent = paper.title;
        document.getElementById("panelAuthors").textContent = authorStr || "Unknown";
        document.getElementById("panelDate").textContent = paper.date || "Unknown Date";
        document.getElementById("panelSource").textContent = paper.source || "Unknown";
        document.getElementById("panelLicense").textContent = paper.license || "Standard";
        document.getElementById("panelAbstractText").textContent = paper.text;
        
        const urlBtn = document.getElementById("panelUrl");
        if(paper.url) {
            urlBtn.href = paper.url;
            urlBtn.classList.remove("hidden");
        } else {
            urlBtn.classList.add("hidden");
        }
        
        knowledgePanel.classList.remove("hidden");
    }
});
