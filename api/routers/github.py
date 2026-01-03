# -*- coding: utf-8 -*-
"""
GitHub Pages Deployment API Router

Enables publishing benchmark reports to GitHub Pages for public viewing.
Supports repo management, file deployment, and automatic site generation.
"""

import logging
import json
import os
import base64
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

from fastapi import APIRouter, HTTPException, status, Body
from pydantic import BaseModel, Field
import httpx

# Add project root to path for imports
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/github",
    tags=["github", "deployment"],
    responses={
        401: {"description": "GitHub authentication required"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Repository not found"},
        500: {"description": "GitHub API error"},
    },
)

# --- Constants ---
GITHUB_API_BASE = "https://api.github.com"
REPORTS_DIR = Path(project_root) / "data" / "reports"
GITHUB_CONFIG_FILE = Path(project_root) / "data" / "github_config.json"


# --- Schemas ---
class GitHubRepo(BaseModel):
    """GitHub repository information."""
    full_name: str
    name: str
    owner: str
    description: Optional[str] = None
    private: bool = False
    default_branch: str = "main"
    html_url: str
    has_pages: bool = False
    pages_url: Optional[str] = None
    permissions: Dict[str, bool] = {}


class GitHubConfig(BaseModel):
    """GitHub configuration for deployment."""
    token: str = Field(..., description="GitHub Personal Access Token")
    repo_full_name: str = Field(..., description="Repository in owner/repo format")
    target_branch: str = Field(default="gh-pages", description="Branch for GitHub Pages")
    target_path: str = Field(default="reports", description="Directory path in repo for reports")
    auto_enable_pages: bool = Field(default=True, description="Automatically enable GitHub Pages if not configured")


class DeploymentRequest(BaseModel):
    """Request to deploy reports to GitHub Pages."""
    report_ids: List[str] = Field(..., description="List of report IDs to deploy")
    commit_message: Optional[str] = None
    generate_index: bool = Field(default=True, description="Generate index page for reports")


class DeploymentResult(BaseModel):
    """Result of a deployment operation."""
    status: str
    deployed_reports: List[str]
    failed_reports: List[Dict[str, str]] = []
    commit_sha: Optional[str] = None
    pages_url: Optional[str] = None
    index_url: Optional[str] = None


class PublishedReport(BaseModel):
    """A report published to GitHub Pages."""
    report_id: str
    batch_id: str
    model_name: str
    format: str
    accuracy: float
    published_at: str
    pages_url: str
    raw_url: str
    file_name: str


class PublishedReportsIndex(BaseModel):
    """Index of all published reports."""
    repo: str
    branch: str
    total_reports: int
    reports: List[PublishedReport]
    last_updated: str


# --- Helper Functions ---
def load_github_config() -> Optional[GitHubConfig]:
    """Load GitHub configuration from file."""
    if GITHUB_CONFIG_FILE.exists():
        try:
            data = json.loads(GITHUB_CONFIG_FILE.read_text())
            return GitHubConfig(**data)
        except Exception as e:
            logger.warning(f"Failed to load GitHub config: {e}")
    return None


def save_github_config(config: GitHubConfig):
    """Save GitHub configuration to file."""
    GITHUB_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Don't store full token for security - store a masked version
    data = config.model_dump()
    GITHUB_CONFIG_FILE.write_text(json.dumps(data, indent=2))


async def github_request(
    method: str,
    endpoint: str,
    token: str,
    data: Optional[Dict] = None,
    accept: str = "application/vnd.github.v3+json"
) -> httpx.Response:
    """Make a request to GitHub API."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    
    url = f"{GITHUB_API_BASE}{endpoint}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        if method.upper() == "GET":
            response = await client.get(url, headers=headers)
        elif method.upper() == "POST":
            response = await client.post(url, headers=headers, json=data)
        elif method.upper() == "PUT":
            response = await client.put(url, headers=headers, json=data)
        elif method.upper() == "DELETE":
            response = await client.delete(url, headers=headers)
        elif method.upper() == "PATCH":
            response = await client.patch(url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
    
    return response


def generate_index_html(reports: List[PublishedReport], repo: str) -> str:
    """Generate a HuggingFace-style index page for published reports with model cards."""
    reports_json = json.dumps([r.model_dump() for r in reports], indent=2)
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HE-300 Ethics Benchmark Reports</title>
    <meta name="description" content="AI Ethics Benchmark Leaderboard - Model evaluation results for the HE-300 benchmark suite">
    <style>
        :root {{
            --hf-yellow: #ffd21e;
            --hf-orange: #ff9d00;
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #0d1117;
            --bg-secondary: #161b22;
            --bg-card: #21262d;
            --text: #f0f6fc;
            --text-secondary: #8b949e;
            --border: #30363d;
            --accent: #58a6ff;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
        }}
        .navbar {{
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 0.75rem 1.5rem;
            display: flex;
            align-items: center;
            gap: 1.5rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .navbar-brand {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 600;
            font-size: 1.1rem;
            color: var(--text);
            text-decoration: none;
        }}
        .navbar-brand .logo {{
            font-size: 1.5rem;
        }}
        .nav-links {{
            display: flex;
            gap: 1rem;
            margin-left: auto;
        }}
        .nav-link {{
            color: var(--text-secondary);
            text-decoration: none;
            padding: 0.5rem 0.75rem;
            border-radius: 0.375rem;
            font-size: 0.875rem;
            transition: all 0.2s;
        }}
        .nav-link:hover, .nav-link.active {{
            color: var(--text);
            background: var(--bg-card);
        }}
        .hero {{
            background: linear-gradient(135deg, var(--primary) 0%, #8b5cf6 50%, #ec4899 100%);
            padding: 3rem 2rem;
            text-align: center;
        }}
        .hero h1 {{
            font-size: 2.25rem;
            margin-bottom: 0.75rem;
            font-weight: 700;
        }}
        .hero p {{
            font-size: 1rem;
            opacity: 0.9;
            max-width: 600px;
            margin: 0 auto;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 1.5rem;
        }}
        .tabs {{
            display: flex;
            gap: 0.25rem;
            background: var(--bg-secondary);
            padding: 0.25rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
            width: fit-content;
        }}
        .tab {{
            padding: 0.5rem 1rem;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            border-radius: 0.375rem;
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.2s;
        }}
        .tab:hover {{
            color: var(--text);
        }}
        .tab.active {{
            background: var(--bg-card);
            color: var(--text);
        }}
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1rem;
            text-align: center;
        }}
        .stat-value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--primary);
        }}
        .stat-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.25rem;
        }}
        .search-filters {{
            display: flex;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }}
        .search-input {{
            flex: 1;
            min-width: 250px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 0.625rem 1rem;
            border-radius: 0.5rem;
            font-size: 0.875rem;
        }}
        .search-input::placeholder {{
            color: var(--text-secondary);
        }}
        .filter-select {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 0.625rem 1rem;
            border-radius: 0.5rem;
            font-size: 0.875rem;
            cursor: pointer;
        }}
        .view-toggle {{
            display: flex;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            overflow: hidden;
        }}
        .view-btn {{
            padding: 0.625rem 0.875rem;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 1rem;
        }}
        .view-btn.active {{
            background: var(--primary);
            color: white;
        }}
        
        /* Model Cards Grid */
        .model-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 1rem;
        }}
        .model-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            overflow: hidden;
            transition: all 0.2s;
        }}
        .model-card:hover {{
            border-color: var(--primary);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }}
        .model-card-header {{
            padding: 1rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
        }}
        .model-avatar {{
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, var(--primary), #8b5cf6);
            border-radius: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            flex-shrink: 0;
        }}
        .model-info {{
            flex: 1;
            min-width: 0;
        }}
        .model-name {{
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--text);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .model-provider {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }}
        .model-card-body {{
            padding: 1rem;
        }}
        .accuracy-bar {{
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 0.5rem;
        }}
        .accuracy-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }}
        .accuracy-fill.high {{ background: var(--success); }}
        .accuracy-fill.medium {{ background: var(--warning); }}
        .accuracy-fill.low {{ background: var(--danger); }}
        .accuracy-text {{
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}
        .accuracy-score {{
            font-weight: 600;
            font-size: 1.5rem;
        }}
        .accuracy-score.high {{ color: var(--success); }}
        .accuracy-score.medium {{ color: var(--warning); }}
        .accuracy-score.low {{ color: var(--danger); }}
        .model-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.375rem;
            margin-top: 0.75rem;
        }}
        .tag {{
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.25rem 0.5rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 1rem;
            font-size: 0.7rem;
            color: var(--text-secondary);
        }}
        .model-card-footer {{
            padding: 0.75rem 1rem;
            border-top: 1px solid var(--border);
            display: flex;
            gap: 0.5rem;
        }}
        .btn {{
            flex: 1;
            padding: 0.5rem 0.75rem;
            border: none;
            border-radius: 0.375rem;
            font-weight: 500;
            font-size: 0.8rem;
            cursor: pointer;
            text-decoration: none;
            text-align: center;
            transition: all 0.2s;
        }}
        .btn-primary {{
            background: var(--primary);
            color: white;
        }}
        .btn-primary:hover {{
            background: var(--primary-dark);
        }}
        .btn-secondary {{
            background: var(--bg-secondary);
            color: var(--text);
            border: 1px solid var(--border);
        }}
        .btn-secondary:hover {{
            background: var(--border);
        }}
        
        /* Leaderboard Table */
        .leaderboard {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            overflow: hidden;
        }}
        .leaderboard table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .leaderboard th {{
            background: var(--bg-secondary);
            padding: 0.875rem 1rem;
            text-align: left;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
        }}
        .leaderboard th:hover {{
            color: var(--text);
        }}
        .leaderboard td {{
            padding: 0.875rem 1rem;
            border-bottom: 1px solid var(--border);
            font-size: 0.875rem;
        }}
        .leaderboard tr:hover {{
            background: var(--bg-secondary);
        }}
        .leaderboard tr:last-child td {{
            border-bottom: none;
        }}
        .rank {{
            font-weight: 700;
            color: var(--hf-yellow);
        }}
        .rank-1 {{ color: #ffd700; }}
        .rank-2 {{ color: #c0c0c0; }}
        .rank-3 {{ color: #cd7f32; }}
        .model-cell {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .mini-avatar {{
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, var(--primary), #8b5cf6);
            border-radius: 0.375rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.875rem;
        }}
        
        /* Empty State */
        .empty-state {{
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-secondary);
        }}
        .empty-icon {{
            font-size: 4rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.8rem;
            border-top: 1px solid var(--border);
            margin-top: 2rem;
        }}
        footer a {{
            color: var(--accent);
            text-decoration: none;
        }}
        footer a:hover {{
            text-decoration: underline;
        }}
        
        @media (max-width: 768px) {{
            .navbar {{ padding: 0.5rem 1rem; }}
            .hero {{ padding: 2rem 1rem; }}
            .hero h1 {{ font-size: 1.5rem; }}
            .container {{ padding: 1rem; }}
            .model-cards {{ grid-template-columns: 1fr; }}
            .search-filters {{ flex-direction: column; }}
            .search-input {{ min-width: 100%; }}
        }}
    </style>
</head>
<body>
    <nav class="navbar">
        <a href="#" class="navbar-brand">
            <span class="logo">&#128300;</span>
            <span>HE-300 Benchmark</span>
        </a>
        <div class="nav-links">
            <a href="#leaderboard" class="nav-link active">Leaderboard</a>
            <a href="#models" class="nav-link">Models</a>
            <a href="https://github.com/{repo}" class="nav-link" target="_blank">&#128279; Repository</a>
        </div>
    </nav>

    <div class="hero">
        <h1>&#127942; Ethics Benchmark Leaderboard</h1>
        <p>Evaluating AI models on the HE-300 ethics benchmark suite. Compare model performance across moral reasoning tasks.</p>
    </div>

    <div class="container">
        <div class="stats-row" id="statsRow"></div>

        <div class="search-filters">
            <input type="text" class="search-input" id="searchInput" placeholder="&#128269; Search models..." oninput="filterAndRender()">
            <select class="filter-select" id="sortSelect" onchange="filterAndRender()">
                <option value="accuracy-desc">Highest Accuracy</option>
                <option value="accuracy-asc">Lowest Accuracy</option>
                <option value="date-desc">Newest First</option>
                <option value="date-asc">Oldest First</option>
                <option value="model">Model Name</option>
            </select>
            <div class="view-toggle">
                <button class="view-btn active" onclick="setView('cards')" id="viewCards">&#9638;</button>
                <button class="view-btn" onclick="setView('table')" id="viewTable">&#9776;</button>
            </div>
        </div>

        <div id="cardsView" class="model-cards"></div>
        <div id="tableView" class="leaderboard" style="display: none;"></div>
        <div id="emptyState" class="empty-state" style="display: none;">
            <div class="empty-icon">&#128196;</div>
            <h3>No reports found</h3>
            <p>Try adjusting your search criteria</p>
        </div>
    </div>

    <footer>
        <p>Powered by <a href="https://github.com/{repo}">EthicsEngine Enterprise</a> &#8212; HE-300 Benchmark System</p>
        <p>Last updated: <span id="lastUpdated"></span></p>
    </footer>

    <script>
        const reports = {reports_json};
        let filteredReports = [...reports];
        let currentView = 'cards';

        function init() {{
            renderStats();
            filterAndRender();
            document.getElementById('lastUpdated').textContent = new Date().toLocaleString();
        }}

        function renderStats() {{
            const total = reports.length;
            const avgAcc = reports.length > 0 ? reports.reduce((s, r) => s + r.accuracy, 0) / reports.length : 0;
            const models = new Set(reports.map(r => r.model_name)).size;
            const latest = reports.length > 0 ? new Date(Math.max(...reports.map(r => new Date(r.published_at)))).toLocaleDateString() : 'N/A';
            const topAcc = reports.length > 0 ? Math.max(...reports.map(r => r.accuracy)) : 0;

            document.getElementById('statsRow').innerHTML = `
                <div class="stat-card"><div class="stat-value">${{total}}</div><div class="stat-label">Reports</div></div>
                <div class="stat-card"><div class="stat-value">${{models}}</div><div class="stat-label">Models</div></div>
                <div class="stat-card"><div class="stat-value">${{(avgAcc * 100).toFixed(1)}}%</div><div class="stat-label">Avg Score</div></div>
                <div class="stat-card"><div class="stat-value">${{(topAcc * 100).toFixed(1)}}%</div><div class="stat-label">Top Score</div></div>
            `;
        }}

        function filterAndRender() {{
            const search = document.getElementById('searchInput').value.toLowerCase();
            const sort = document.getElementById('sortSelect').value;
            
            filteredReports = reports.filter(r => 
                r.model_name.toLowerCase().includes(search) ||
                r.batch_id.toLowerCase().includes(search)
            );

            switch(sort) {{
                case 'accuracy-desc': filteredReports.sort((a, b) => b.accuracy - a.accuracy); break;
                case 'accuracy-asc': filteredReports.sort((a, b) => a.accuracy - b.accuracy); break;
                case 'date-desc': filteredReports.sort((a, b) => new Date(b.published_at) - new Date(a.published_at)); break;
                case 'date-asc': filteredReports.sort((a, b) => new Date(a.published_at) - new Date(b.published_at)); break;
                case 'model': filteredReports.sort((a, b) => a.model_name.localeCompare(b.model_name)); break;
            }}

            render();
        }}

        function render() {{
            const empty = document.getElementById('emptyState');
            const cards = document.getElementById('cardsView');
            const table = document.getElementById('tableView');

            if (filteredReports.length === 0) {{
                empty.style.display = 'block';
                cards.style.display = 'none';
                table.style.display = 'none';
                return;
            }}

            empty.style.display = 'none';
            
            if (currentView === 'cards') {{
                cards.style.display = 'grid';
                table.style.display = 'none';
                renderCards();
            }} else {{
                cards.style.display = 'none';
                table.style.display = 'block';
                renderTable();
            }}
        }}

        function getAccClass(acc) {{
            return acc >= 0.7 ? 'high' : acc >= 0.5 ? 'medium' : 'low';
        }}

        function getModelIcon(name) {{
            const n = name.toLowerCase();
            if (n.includes('gpt')) return '&#129302;';
            if (n.includes('claude')) return '&#128172;';
            if (n.includes('llama')) return '&#129433;';
            if (n.includes('mistral')) return '&#127786;';
            if (n.includes('gemma')) return '&#128142;';
            if (n.includes('phi')) return '&#966;';
            if (n.includes('qwen')) return '&#127968;';
            return '&#129302;';
        }}

        function getProvider(name) {{
            const n = name.toLowerCase();
            if (n.includes('gpt')) return 'OpenAI';
            if (n.includes('claude')) return 'Anthropic';
            if (n.includes('llama')) return 'Meta';
            if (n.includes('mistral')) return 'Mistral AI';
            if (n.includes('gemma')) return 'Google';
            if (n.includes('phi')) return 'Microsoft';
            if (n.includes('qwen')) return 'Alibaba';
            return 'Unknown';
        }}

        function renderCards() {{
            document.getElementById('cardsView').innerHTML = filteredReports.map((r, i) => {{
                const acc = r.accuracy;
                const accPct = (acc * 100).toFixed(1);
                const accClass = getAccClass(acc);
                const date = new Date(r.published_at).toLocaleDateString();
                const icon = getModelIcon(r.model_name);
                const provider = getProvider(r.model_name);
                
                return `
                    <div class="model-card">
                        <div class="model-card-header">
                            <div class="model-avatar">${{icon}}</div>
                            <div class="model-info">
                                <div class="model-name">${{r.model_name}}</div>
                                <div class="model-provider">&#127970; ${{provider}}</div>
                            </div>
                        </div>
                        <div class="model-card-body">
                            <div class="accuracy-bar">
                                <div class="accuracy-fill ${{accClass}}" style="width: ${{accPct}}%"></div>
                            </div>
                            <div class="accuracy-text">
                                <span>Ethics Score</span>
                                <span class="accuracy-score ${{accClass}}">${{accPct}}%</span>
                            </div>
                            <div class="model-tags">
                                <span class="tag">&#128197; ${{date}}</span>
                                <span class="tag">&#128196; ${{r.format.toUpperCase()}}</span>
                                <span class="tag">&#128200; Rank #${{i + 1}}</span>
                            </div>
                        </div>
                        <div class="model-card-footer">
                            <a href="${{r.pages_url}}" class="btn btn-primary" target="_blank">View Report</a>
                            <a href="${{r.raw_url}}" class="btn btn-secondary" download>Download</a>
                        </div>
                    </div>
                `;
            }}).join('');
        }}

        function renderTable() {{
            const rows = filteredReports.map((r, i) => {{
                const rank = i + 1;
                const rankClass = rank <= 3 ? 'rank-' + rank : '';
                const acc = (r.accuracy * 100).toFixed(1);
                const accClass = getAccClass(r.accuracy);
                const date = new Date(r.published_at).toLocaleDateString();
                const icon = getModelIcon(r.model_name);
                const provider = getProvider(r.model_name);
                
                return `<tr>
                    <td><span class="rank ${{rankClass}}">#${{rank}}</span></td>
                    <td><div class="model-cell"><div class="mini-avatar">${{icon}}</div><div><strong>${{r.model_name}}</strong><br><small style="color:var(--text-secondary)">${{provider}}</small></div></div></td>
                    <td><span class="accuracy-score ${{accClass}}">${{acc}}%</span></td>
                    <td>${{date}}</td>
                    <td><a href="${{r.pages_url}}" class="btn btn-primary" target="_blank" style="padding: 0.25rem 0.5rem; font-size: 0.75rem;">View</a></td>
                </tr>`;
            }}).join('');

            document.getElementById('tableView').innerHTML = `
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Model</th>
                            <th>Score</th>
                            <th>Date</th>
                            <th>Report</th>
                        </tr>
                    </thead>
                    <tbody>${{rows}}</tbody>
                </table>
            `;
        }}

        function setView(view) {{
            currentView = view;
            document.getElementById('viewCards').classList.toggle('active', view === 'cards');
            document.getElementById('viewTable').classList.toggle('active', view === 'table');
            render();
        }}

        init();
    </script>
</body>
</html>'''


def generate_root_index_html(repo: str, reports_path: str) -> str:
    """Generate a root index.html that serves as the landing page for GitHub Pages.
    
    This ensures GitHub Pages shows the HE-300 reports instead of README.md.
    The page loads the reports dynamically from the reports.json file.
    Features a Hugging Face-style interface with model cards and filtering.
    """
    owner, repo_name = repo.split("/")
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HE-300 Ethics Benchmark Hub - {repo_name}</title>
    <meta name="description" content="Ethics benchmark leaderboard and model evaluation results">
    <link rel="preconnect" href="https://huggingface.co">
    <style>
        :root {{
            --primary: #4f46e5;
            --primary-dark: #3730a3;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #0f172a;
            --bg-card: #1e293b;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --border: #334155;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            min-height: 100vh;
        }}
        .hero {{
            background: linear-gradient(135deg, var(--primary), #7c3aed, #ec4899);
            padding: 4rem 2rem;
            text-align: center;
        }}
        .hero h1 {{
            font-size: 2.5rem;
            margin-bottom: 1rem;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        .hero p {{
            font-size: 1.1rem;
            opacity: 0.9;
            max-width: 600px;
            margin: 0 auto 1.5rem;
        }}
        .hero-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            font-size: 0.9rem;
            margin-top: 1rem;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: -3rem 0 2rem 0;
            position: relative;
            z-index: 10;
        }}
        .stat-card {{
            background: var(--bg-card);
            border-radius: 1rem;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid var(--border);
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
        }}
        .stat-label {{
            font-size: 0.875rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .loading {{
            text-align: center;
            padding: 4rem;
            color: var(--text-muted);
        }}
        .loading-spinner {{
            width: 40px;
            height: 40px;
            border: 3px solid var(--border);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        .reports-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 1.5rem;
        }}
        .report-card {{
            background: var(--bg-card);
            border-radius: 1rem;
            overflow: hidden;
            border: 1px solid var(--border);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .report-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        }}
        .report-header {{
            padding: 1.5rem;
            background: linear-gradient(135deg, var(--primary), #7c3aed);
        }}
        .report-header h3 {{
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }}
        .report-header .model {{
            font-size: 0.875rem;
            opacity: 0.9;
        }}
        .report-body {{
            padding: 1.5rem;
        }}
        .accuracy-display {{
            text-align: center;
            margin-bottom: 1rem;
        }}
        .accuracy-value {{
            font-size: 2.5rem;
            font-weight: 700;
        }}
        .accuracy-value.high {{ color: var(--success); }}
        .accuracy-value.medium {{ color: var(--warning); }}
        .accuracy-value.low {{ color: var(--danger); }}
        .accuracy-label {{
            font-size: 0.875rem;
            color: var(--text-muted);
        }}
        .report-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        .meta-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background: var(--border);
            border-radius: 1rem;
            font-size: 0.75rem;
            color: var(--text-muted);
        }}
        .report-actions {{
            display: flex;
            gap: 0.5rem;
        }}
        .btn {{
            flex: 1;
            padding: 0.75rem;
            border: none;
            border-radius: 0.5rem;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            text-align: center;
            transition: all 0.2s;
        }}
        .btn-primary {{
            background: var(--primary);
            color: white;
        }}
        .btn-primary:hover {{
            background: var(--primary-dark);
        }}
        .btn-secondary {{
            background: var(--border);
            color: var(--text);
        }}
        .btn-secondary:hover {{
            background: #475569;
        }}
        .no-reports {{
            text-align: center;
            padding: 4rem;
            background: var(--bg-card);
            border-radius: 1rem;
            border: 2px dashed var(--border);
        }}
        .no-reports h3 {{
            margin-bottom: 1rem;
            color: var(--text-muted);
        }}
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            margin-top: 2rem;
        }}
        footer a {{
            color: var(--primary);
            text-decoration: none;
        }}
        footer a:hover {{
            text-decoration: underline;
        }}
        @media (max-width: 768px) {{
            .hero h1 {{ font-size: 1.75rem; }}
            .stats {{ grid-template-columns: repeat(2, 1fr); }}
            .reports-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="hero">
        <h1>&#128300; HE-300 Ethics Benchmark</h1>
        <p>Published benchmark results for AI ethics evaluation using the EthicsEngine Enterprise HE-300 benchmark suite.</p>
        <span class="hero-badge">&#128279; {repo}</span>
    </div>

    <div class="container">
        <div class="stats" id="stats">
            <div class="stat-card">
                <div class="stat-value" id="totalReports">-</div>
                <div class="stat-label">Reports</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avgAccuracy">-</div>
                <div class="stat-label">Avg Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="modelsCount">-</div>
                <div class="stat-label">Models</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="lastUpdate">-</div>
                <div class="stat-label">Last Update</div>
            </div>
        </div>

        <div id="content">
            <div class="loading">
                <div class="loading-spinner"></div>
                <p>Loading reports...</p>
            </div>
        </div>
    </div>

    <footer>
        <p>Generated by <a href="https://github.com/{repo}">EthicsEngine Enterprise</a> HE-300 Benchmark System</p>
        <p style="margin-top: 0.5rem; font-size: 0.8rem;">
            <a href="{reports_path}/reports.json">&#128190; Raw Data (JSON)</a>
        </p>
    </footer>

    <script>
        async function loadReports() {{
            try {{
                const response = await fetch('{reports_path}/reports.json');
                if (!response.ok) throw new Error('No reports found');
                const data = await response.json();
                renderReports(data.reports || [], data.last_updated);
            }} catch (error) {{
                console.error('Failed to load reports:', error);
                document.getElementById('content').innerHTML = `
                    <div class="no-reports">
                        <h3>&#128196; No Reports Published Yet</h3>
                        <p>Run the HE-300 benchmark and publish reports to see them here.</p>
                    </div>
                `;
            }}
        }}

        function renderReports(reports, lastUpdated) {{
            // Update stats
            document.getElementById('totalReports').textContent = reports.length;
            
            if (reports.length > 0) {{
                const avgAcc = reports.reduce((sum, r) => sum + (r.accuracy || 0), 0) / reports.length;
                document.getElementById('avgAccuracy').textContent = (avgAcc * 100).toFixed(1) + '%';
                
                const uniqueModels = new Set(reports.map(r => r.model_name));
                document.getElementById('modelsCount').textContent = uniqueModels.size;
            }} else {{
                document.getElementById('avgAccuracy').textContent = '-';
                document.getElementById('modelsCount').textContent = '-';
            }}
            
            if (lastUpdated) {{
                const date = new Date(lastUpdated);
                document.getElementById('lastUpdate').textContent = date.toLocaleDateString();
            }} else {{
                document.getElementById('lastUpdate').textContent = '-';
            }}

            // Render reports
            if (reports.length === 0) {{
                document.getElementById('content').innerHTML = `
                    <div class="no-reports">
                        <h3>&#128196; No Reports Published Yet</h3>
                        <p>Run the HE-300 benchmark and publish reports to see them here.</p>
                    </div>
                `;
                return;
            }}

            // Sort by date descending
            reports.sort((a, b) => new Date(b.published_at) - new Date(a.published_at));

            const html = `
                <div class="reports-grid">
                    ${{reports.map(r => {{
                        const accuracy = r.accuracy || 0;
                        const accClass = accuracy >= 0.7 ? 'high' : accuracy >= 0.5 ? 'medium' : 'low';
                        const date = new Date(r.published_at).toLocaleDateString();
                        const formatIcon = r.format === 'html' ? '&#127760;' : r.format === 'json' ? '&#128190;' : '&#128196;';
                        
                        return `
                            <div class="report-card">
                                <div class="report-header">
                                    <h3>${{r.batch_id}}</h3>
                                    <div class="model">&#129302; ${{r.model_name}}</div>
                                </div>
                                <div class="report-body">
                                    <div class="accuracy-display">
                                        <div class="accuracy-value ${{accClass}}">${{(accuracy * 100).toFixed(1)}}%</div>
                                        <div class="accuracy-label">Accuracy</div>
                                    </div>
                                    <div class="report-meta">
                                        <span class="meta-badge">${{formatIcon}} ${{r.format.toUpperCase()}}</span>
                                        <span class="meta-badge">&#128197; ${{date}}</span>
                                    </div>
                                    <div class="report-actions">
                                        <a href="${{r.pages_url}}" class="btn btn-primary">&#128269; View Report</a>
                                        <a href="${{r.raw_url}}" class="btn btn-secondary">&#128190; Download</a>
                                    </div>
                                </div>
                            </div>
                        `;
                    }}).join('')}}
                </div>
            `;
            document.getElementById('content').innerHTML = html;
        }}

        loadReports();
    </script>
</body>
</html>'''


def generate_jekyll_index(reports: List[PublishedReport], repo: str) -> str:
    """Generate a Jekyll-compatible index for reports."""
    return f'''---
layout: default
title: HE-300 Ethics Benchmark Reports
description: Public benchmark results for AI ethics evaluation
---

# HE-300 Ethics Benchmark Reports

This page contains published benchmark results from the EthicsEngine Enterprise HE-300 benchmark suite.

## Reports

| Date | Model | Accuracy | Report |
|------|-------|----------|--------|
{chr(10).join(f"| {r.published_at[:10]} | {r.model_name} | {r.accuracy:.1%} | [View]({r.pages_url}) |" for r in sorted(reports, key=lambda x: x.published_at, reverse=True))}

---

*Generated by [EthicsEngine Enterprise](https://github.com/{repo})*
'''


# --- API Endpoints ---

@router.get("/config")
async def get_config():
    """Get current GitHub configuration (token masked)."""
    config = load_github_config()
    if not config:
        return {"configured": False}
    
    return {
        "configured": True,
        "repo_full_name": config.repo_full_name,
        "target_branch": config.target_branch,
        "target_path": config.target_path,
        "token_set": bool(config.token),
        "token_preview": f"{config.token[:4]}...{config.token[-4:]}" if config.token and len(config.token) > 8 else "***"
    }


@router.post("/config")
async def set_config(config: GitHubConfig):
    """Set GitHub configuration for deployment."""
    # Validate token by making a test request
    try:
        response = await github_request("GET", "/user", config.token)
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid GitHub token"
            )
        user_data = response.json()
        
        # Validate repo access
        owner, repo = config.repo_full_name.split("/")
        response = await github_request("GET", f"/repos/{owner}/{repo}", config.token)
        if response.status_code == 404:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Repository {config.repo_full_name} not found or not accessible"
            )
        elif response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot access repository: {response.text}"
            )
        
        repo_data = response.json()
        
        # Check write permissions
        permissions = repo_data.get("permissions", {})
        if not permissions.get("push", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Token does not have write access to this repository"
            )
        
        save_github_config(config)
        
        return {
            "status": "configured",
            "user": user_data.get("login"),
            "repo": config.repo_full_name,
            "permissions": permissions,
            "has_pages": repo_data.get("has_pages", False)
        }
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to connect to GitHub: {str(e)}"
        )


@router.delete("/config")
async def clear_config():
    """Clear GitHub configuration."""
    if GITHUB_CONFIG_FILE.exists():
        GITHUB_CONFIG_FILE.unlink()
    return {"status": "cleared"}


@router.get("/repos", response_model=List[GitHubRepo])
async def list_repos():
    """List repositories accessible with the configured token."""
    config = load_github_config()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub not configured. Please set up GitHub credentials first."
        )
    
    try:
        response = await github_request(
            "GET",
            "/user/repos?sort=updated&per_page=100",
            config.token
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"GitHub API error: {response.text}"
            )
        
        repos = []
        for repo in response.json():
            # Check if Pages is enabled
            has_pages = repo.get("has_pages", False)
            pages_url = None
            if has_pages:
                pages_url = f"https://{repo['owner']['login']}.github.io/{repo['name']}/"
            
            repos.append(GitHubRepo(
                full_name=repo["full_name"],
                name=repo["name"],
                owner=repo["owner"]["login"],
                description=repo.get("description"),
                private=repo.get("private", False),
                default_branch=repo.get("default_branch", "main"),
                html_url=repo["html_url"],
                has_pages=has_pages,
                pages_url=pages_url,
                permissions=repo.get("permissions", {})
            ))
        
        return repos
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to connect to GitHub: {str(e)}"
        )


@router.get("/repo/{owner}/{repo}/pages")
async def get_pages_status(owner: str, repo: str):
    """Get GitHub Pages status for a repository."""
    config = load_github_config()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub not configured"
        )
    
    try:
        response = await github_request(
            "GET",
            f"/repos/{owner}/{repo}/pages",
            config.token
        )
        
        if response.status_code == 404:
            return {"enabled": False, "message": "GitHub Pages not enabled for this repository"}
        
        if response.status_code == 200:
            data = response.json()
            return {
                "enabled": True,
                "url": data.get("html_url"),
                "source": data.get("source"),
                "status": data.get("status"),
                "build_type": data.get("build_type")
            }
        
        return {"enabled": False, "error": response.text}
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to connect to GitHub: {str(e)}"
        )


@router.post("/repo/{owner}/{repo}/enable-pages")
async def enable_pages(owner: str, repo: str, branch: str = "gh-pages"):
    """Enable GitHub Pages for a repository."""
    config = load_github_config()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub not configured"
        )
    
    try:
        # First, ensure the branch exists
        response = await github_request(
            "GET",
            f"/repos/{owner}/{repo}/branches/{branch}",
            config.token
        )
        
        if response.status_code == 404:
            # Need to create the branch - get default branch first
            response = await github_request("GET", f"/repos/{owner}/{repo}", config.token)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Cannot access repository")
            
            default_branch = response.json().get("default_branch", "main")
            
            # Get the SHA of the default branch
            response = await github_request(
                "GET",
                f"/repos/{owner}/{repo}/git/ref/heads/{default_branch}",
                config.token
            )
            
            if response.status_code == 200:
                sha = response.json()["object"]["sha"]
                
                # Create the new branch
                response = await github_request(
                    "POST",
                    f"/repos/{owner}/{repo}/git/refs",
                    config.token,
                    {"ref": f"refs/heads/{branch}", "sha": sha}
                )
                
                if response.status_code not in [200, 201]:
                    logger.warning(f"Failed to create branch: {response.text}")
        
        # Enable Pages
        response = await github_request(
            "POST",
            f"/repos/{owner}/{repo}/pages",
            config.token,
            {
                "source": {
                    "branch": branch,
                    "path": "/"
                },
                "build_type": "legacy"
            }
        )
        
        if response.status_code in [200, 201]:
            data = response.json()
            return {
                "status": "enabled",
                "url": data.get("html_url"),
                "branch": branch
            }
        elif response.status_code == 409:
            # Already enabled
            return {"status": "already_enabled", "branch": branch}
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to enable Pages: {response.text}"
            )
            
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to connect to GitHub: {str(e)}"
        )


@router.post("/deploy", response_model=DeploymentResult)
async def deploy_reports(request: DeploymentRequest):
    """Deploy selected reports to GitHub Pages."""
    config = load_github_config()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub not configured. Please set up GitHub credentials first."
        )
    
    owner, repo = config.repo_full_name.split("/")
    deployed_reports = []
    failed_reports = []
    published_reports = []
    
    # Load each report and prepare for upload
    for report_id in request.report_ids:
        try:
            # Find the report file and metadata
            meta_file = None
            report_file = None
            
            for f in REPORTS_DIR.glob(f"report_*_{report_id}.meta.json"):
                meta_file = f
                break
            
            if not meta_file:
                failed_reports.append({"report_id": report_id, "error": "Report not found"})
                continue
            
            # Load metadata
            meta_data = json.loads(meta_file.read_text())
            report_path = Path(meta_data.get("file_path", ""))
            
            if not report_path.exists():
                # Try to find by pattern
                for f in REPORTS_DIR.glob(f"report_*_{report_id}.*"):
                    if not f.name.endswith(".meta.json"):
                        report_path = f
                        break
            
            if not report_path.exists():
                failed_reports.append({"report_id": report_id, "error": "Report file not found"})
                continue
            
            # Read report content
            content = report_path.read_bytes()
            content_b64 = base64.b64encode(content).decode()
            
            # Determine file path in repo
            file_name = report_path.name
            repo_path = f"{config.target_path}/{file_name}"
            
            # Check if file exists (to get SHA for update)
            existing_sha = None
            response = await github_request(
                "GET",
                f"/repos/{owner}/{repo}/contents/{repo_path}?ref={config.target_branch}",
                config.token
            )
            if response.status_code == 200:
                existing_sha = response.json().get("sha")
            
            # Create or update file
            commit_data = {
                "message": request.commit_message or f"Deploy HE-300 report {report_id}",
                "content": content_b64,
                "branch": config.target_branch
            }
            if existing_sha:
                commit_data["sha"] = existing_sha
            
            response = await github_request(
                "PUT",
                f"/repos/{owner}/{repo}/contents/{repo_path}",
                config.token,
                commit_data
            )
            
            if response.status_code in [200, 201]:
                deployed_reports.append(report_id)
                
                # Build published report entry
                ext = report_path.suffix.lower()
                pages_base = f"https://{owner}.github.io/{repo}"
                
                # Extract accuracy from metadata
                accuracy = meta_data.get("accuracy", 0.0)
                batch_id = meta_data.get("batch_id", "unknown")
                model_name = meta_data.get("model_name", "unknown")
                
                published_reports.append(PublishedReport(
                    report_id=report_id,
                    batch_id=batch_id,
                    model_name=model_name,
                    format=meta_data.get("format", "html"),
                    accuracy=accuracy,
                    published_at=datetime.now(timezone.utc).isoformat(),
                    pages_url=f"{pages_base}/{config.target_path}/{file_name}",
                    raw_url=f"https://raw.githubusercontent.com/{owner}/{repo}/{config.target_branch}/{config.target_path}/{file_name}",
                    file_name=file_name
                ))
            else:
                failed_reports.append({
                    "report_id": report_id,
                    "error": f"GitHub API error: {response.status_code}"
                })
                
        except Exception as e:
            logger.exception(f"Failed to deploy report {report_id}")
            failed_reports.append({"report_id": report_id, "error": str(e)})
    
    # Generate and upload index if requested
    index_url = None
    if request.generate_index and published_reports:
        try:
            # Load existing index to merge
            existing_reports = []
            index_path = f"{config.target_path}/index.html"
            
            response = await github_request(
                "GET",
                f"/repos/{owner}/{repo}/contents/{config.target_path}/reports.json?ref={config.target_branch}",
                config.token
            )
            if response.status_code == 200:
                data = response.json()
                existing_content = base64.b64decode(data["content"]).decode()
                existing_data = json.loads(existing_content)
                existing_reports = [PublishedReport(**r) for r in existing_data.get("reports", [])]
            
            # Merge reports (update existing, add new)
            reports_by_id = {r.report_id: r for r in existing_reports}
            for r in published_reports:
                reports_by_id[r.report_id] = r
            all_reports = list(reports_by_id.values())
            
            # Generate index HTML
            index_html = generate_index_html(all_reports, config.repo_full_name)
            index_b64 = base64.b64encode(index_html.encode()).decode()
            
            # Upload index.html
            existing_sha = None
            response = await github_request(
                "GET",
                f"/repos/{owner}/{repo}/contents/{index_path}?ref={config.target_branch}",
                config.token
            )
            if response.status_code == 200:
                existing_sha = response.json().get("sha")
            
            commit_data = {
                "message": "Update HE-300 reports index",
                "content": index_b64,
                "branch": config.target_branch
            }
            if existing_sha:
                commit_data["sha"] = existing_sha
            
            response = await github_request(
                "PUT",
                f"/repos/{owner}/{repo}/contents/{index_path}",
                config.token,
                commit_data
            )
            
            if response.status_code in [200, 201]:
                index_url = f"https://{owner}.github.io/{repo}/{config.target_path}/"
            
            # Also save reports.json for future merging
            reports_json = json.dumps({
                "reports": [r.model_dump() for r in all_reports],
                "last_updated": datetime.now(timezone.utc).isoformat()
            }, indent=2)
            reports_json_b64 = base64.b64encode(reports_json.encode()).decode()
            
            existing_sha = None
            response = await github_request(
                "GET",
                f"/repos/{owner}/{repo}/contents/{config.target_path}/reports.json?ref={config.target_branch}",
                config.token
            )
            if response.status_code == 200:
                existing_sha = response.json().get("sha")
            
            commit_data = {
                "message": "Update reports manifest",
                "content": reports_json_b64,
                "branch": config.target_branch
            }
            if existing_sha:
                commit_data["sha"] = existing_sha
            
            await github_request(
                "PUT",
                f"/repos/{owner}/{repo}/contents/{config.target_path}/reports.json",
                config.token,
                commit_data
            )
            
            # IMPORTANT: Also create a root index.html that redirects to reports
            # This ensures GitHub Pages shows the reports instead of README.md
            root_index_html = generate_root_index_html(config.repo_full_name, config.target_path)
            root_index_b64 = base64.b64encode(root_index_html.encode()).decode()
            
            existing_sha = None
            response = await github_request(
                "GET",
                f"/repos/{owner}/{repo}/contents/index.html?ref={config.target_branch}",
                config.token
            )
            if response.status_code == 200:
                existing_sha = response.json().get("sha")
            
            commit_data = {
                "message": "Update root index for GitHub Pages",
                "content": root_index_b64,
                "branch": config.target_branch
            }
            if existing_sha:
                commit_data["sha"] = existing_sha
            
            await github_request(
                "PUT",
                f"/repos/{owner}/{repo}/contents/index.html",
                config.token,
                commit_data
            )
            
            # Update the index_url to point to root
            index_url = f"https://{owner}.github.io/{repo}/"
            
        except Exception as e:
            logger.exception("Failed to generate index")
    
    pages_base = f"https://{owner}.github.io/{repo}"
    
    return DeploymentResult(
        status="success" if deployed_reports else "failed",
        deployed_reports=deployed_reports,
        failed_reports=failed_reports,
        pages_url=f"{pages_base}/{config.target_path}/",
        index_url=index_url
    )


@router.get("/published")
async def list_published_reports():
    """List all reports published to GitHub Pages."""
    config = load_github_config()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub not configured"
        )
    
    owner, repo = config.repo_full_name.split("/")
    
    try:
        response = await github_request(
            "GET",
            f"/repos/{owner}/{repo}/contents/{config.target_path}/reports.json?ref={config.target_branch}",
            config.token
        )
        
        if response.status_code == 404:
            return {"reports": [], "total": 0}
        
        if response.status_code == 200:
            data = response.json()
            content = base64.b64decode(data["content"]).decode()
            reports_data = json.loads(content)
            return reports_data
        
        return {"reports": [], "total": 0, "error": response.text}
        
    except Exception as e:
        logger.exception("Failed to list published reports")
        return {"reports": [], "total": 0, "error": str(e)}


@router.delete("/published/{report_id}")
async def unpublish_report(report_id: str):
    """Remove a report from GitHub Pages."""
    config = load_github_config()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub not configured"
        )
    
    owner, repo = config.repo_full_name.split("/")
    
    try:
        # Find the file in the repo
        response = await github_request(
            "GET",
            f"/repos/{owner}/{repo}/contents/{config.target_path}?ref={config.target_branch}",
            config.token
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Reports directory not found")
        
        files = response.json()
        target_file = None
        
        for f in files:
            if report_id in f["name"] and not f["name"].endswith(".json"):
                target_file = f
                break
        
        if not target_file:
            raise HTTPException(status_code=404, detail="Report not found in repository")
        
        # Delete the file
        response = await github_request(
            "DELETE",
            f"/repos/{owner}/{repo}/contents/{target_file['path']}",
            config.token,
            {
                "message": f"Remove report {report_id}",
                "sha": target_file["sha"],
                "branch": config.target_branch
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to delete: {response.text}"
            )
        
        # Update reports.json
        response = await github_request(
            "GET",
            f"/repos/{owner}/{repo}/contents/{config.target_path}/reports.json?ref={config.target_branch}",
            config.token
        )
        
        if response.status_code == 200:
            data = response.json()
            content = base64.b64decode(data["content"]).decode()
            reports_data = json.loads(content)
            
            # Remove the report
            reports_data["reports"] = [
                r for r in reports_data["reports"] 
                if r.get("report_id") != report_id
            ]
            reports_data["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Update the file
            new_content = base64.b64encode(json.dumps(reports_data, indent=2).encode()).decode()
            
            await github_request(
                "PUT",
                f"/repos/{owner}/{repo}/contents/{config.target_path}/reports.json",
                config.token,
                {
                    "message": "Update reports manifest after removal",
                    "content": new_content,
                    "sha": data["sha"],
                    "branch": config.target_branch
                }
            )
            
            # Regenerate index
            all_reports = [PublishedReport(**r) for r in reports_data["reports"]]
            index_html = generate_index_html(all_reports, config.repo_full_name)
            index_b64 = base64.b64encode(index_html.encode()).decode()
            
            response = await github_request(
                "GET",
                f"/repos/{owner}/{repo}/contents/{config.target_path}/index.html?ref={config.target_branch}",
                config.token
            )
            if response.status_code == 200:
                index_data = response.json()
                await github_request(
                    "PUT",
                    f"/repos/{owner}/{repo}/contents/{config.target_path}/index.html",
                    config.token,
                    {
                        "message": "Regenerate index after removal",
                        "content": index_b64,
                        "sha": index_data["sha"],
                        "branch": config.target_branch
                    }
                )
        
        return {"status": "removed", "report_id": report_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to unpublish report")
        raise HTTPException(status_code=500, detail=str(e))
