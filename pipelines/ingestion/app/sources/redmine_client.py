"""Redmine API client for fetching FAQ issues."""

import logging
from datetime import datetime
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


# Import from shared utilities
import sys
sys.path.insert(0, "/home/phungkien/ehc_ai_mvp")

from shared.py.models.faq import FAQSource


class RedmineClient:
    """Fetch FAQ from Redmine API."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"X-Redmine-API-Key": api_key})
    
    def fetch_faq_issues(self, project_key: str = "FAQ", status_ids: List[str] = None) -> List[FAQSource]:
        """
        Fetch all FAQ issues from a Redmine project.
        
        Args:
            project_key: Redmine project identifier (default "FAQ")
            status_ids: Issue status IDs to fetch (default "*" = all)
        
        Returns:
            List of FAQSource objects
        """
        
        if status_ids is None:
            status_ids = ["*"]  # All statuses
        
        issues = []
        offset = 0
        limit = 100  # Redmine API page size
        
        try:
            while True:
                url = f"{self.base_url}/projects/{project_key}/issues.json"
                params = {
                    "offset": offset,
                    "limit": limit,
                    "status_id": ",".join(status_ids),
                    # Redmine expects a comma-separated string for include.
                    # Passing a list becomes repeated query params and can drop attachments.
                    "include": "attachments,custom_fields",
                }
                
                logger.info(f"Fetching issues from Redmine: offset={offset}, limit={limit}")
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                for issue_data in data.get("issues", []):
                    faq = self._parse_issue(issue_data)
                    if faq:
                        issues.append(faq)
                
                total_count = data.get("total_count", 0)
                offset += limit
                
                if offset >= total_count:
                    break
            
            logger.info(f"Fetched {len(issues)} FAQ issues from Redmine")
            return issues
        
        except Exception as e:
            logger.error(f"Failed to fetch Redmine issues: {e}")
            return []
    
    def _parse_issue(self, issue_dict: dict) -> Optional[FAQSource]:
        """Parse Redmine issue JSON into FAQSource."""
        try:
            issue_id = str(issue_dict.get("id", ""))
            subject = issue_dict.get("subject", "")
            description = issue_dict.get("description", "")
            
            # Extract attachment URLs
            attachments = issue_dict.get("attachments", [])
            attachment_urls = [att.get("content_url", "") for att in attachments if att.get("content_url")]
            
            # Custom fields
            custom_fields_list = issue_dict.get("custom_fields", [])
            custom_fields = {cf.get("name", ""): cf.get("value", "") for cf in custom_fields_list}
            
            # Timestamps
            created_at = datetime.fromisoformat(issue_dict.get("created_on", "").replace("Z", "+00:00"))
            updated_at = datetime.fromisoformat(issue_dict.get("updated_on", "").replace("Z", "+00:00"))
            
            url = f"{self.base_url}/issues/{issue_id}"
            
            return FAQSource(
                issue_id=issue_id,
                subject=subject,
                description=description,
                custom_fields=custom_fields,
                attachment_urls=attachment_urls,
                url=url,
                created_at=created_at,
                updated_at=updated_at
            )
        except Exception as e:
            logger.error(f"Failed to parse issue: {e}")
            return None
