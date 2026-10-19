import streamlit as st
import requests
import os

# --- Workflow Trigger Function ---
def trigger_github_workflow():
    # Load secrets from Streamlit secrets management or environment
    try:
        token = st.secrets["GITHUB_TOKEN"]
        owner = st.secrets["REPO_OWNER"]
        repo = st.secrets["REPO_NAME"]
    except Exception:
        st.error("⚠️ GitHub secrets (GITHUB_TOKEN, REPO_OWNER, REPO_NAME) are not configured in Streamlit settings.")
        return False

    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/pipeline.yml/dispatches"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    # workflow_dispatch requires the branch name
    payload = {
        "ref": "main"  # or "master" depending on your default branch
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 204:
        return True
    else:
        st.error(f"Failed to trigger workflow: {response.text}")
        return False

# --- Add to Sidebar UI ---
st.sidebar.divider()
st.sidebar.subheader("⚙️ Pipeline Controls")
if st.sidebar.button("🚀 Run Workflow Now"):
    with st.spinner("Triggering GitHub Actions workflow..."):
        success = trigger_github_workflow()
        if success:
            st.sidebar.success("Workflow triggered successfully! Check GitHub Actions for progress.")
