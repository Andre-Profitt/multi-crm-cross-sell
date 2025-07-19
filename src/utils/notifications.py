import asyncio
import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)


class NotificationManager:
    """Manages notifications across different channels"""

    def __init__(self):
        self.smtp_config = {
            "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
            "port": int(os.getenv("SMTP_PORT", "587")),
            "username": os.getenv("SMTP_USERNAME"),
            "password": os.getenv("SMTP_PASSWORD"),
            "from_email": os.getenv("NOTIFICATION_FROM", "noreply@crosssell.ai"),
        }

        self.slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        self.teams_webhook = os.getenv("TEAMS_WEBHOOK_URL")

        self.notification_channels = []
        if self.smtp_config["username"]:
            self.notification_channels.append("email")
        if self.slack_webhook:
            self.notification_channels.append("slack")
        if self.teams_webhook:
            self.notification_channels.append("teams")

    async def send_opportunity_alert(self, opportunities: pd.DataFrame):
        """Send alert for high-value opportunities"""
        if opportunities.empty:
            return

        subject = f"🎯 {len(opportunities)} New High-Value Cross-Sell Opportunities"

        # Create HTML content
        html_content = self._create_opportunity_html(opportunities)

        # Create plain text content
        text_content = self._create_opportunity_text(opportunities)

        # Send via all configured channels
        tasks = []

        if "email" in self.notification_channels:
            recipients = os.getenv("NOTIFICATION_RECIPIENTS", "").split(",")
            for recipient in recipients:
                if recipient:
                    tasks.append(
                        self._send_email(recipient.strip(), subject, html_content, text_content)
                    )

        if "slack" in self.notification_channels:
            tasks.append(self._send_slack(subject, opportunities))

        if "teams" in self.notification_channels:
            tasks.append(self._send_teams(subject, opportunities))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def send_error_notification(self, error_message: str):
        """Send error notification to admins"""
        subject = "❌ Cross-Sell Pipeline Error"

        html_content = f"""
        <html>
            <body>
                <h2 style="color: #d32f2f;">Pipeline Error Detected</h2>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Error:</strong></p>
                <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
{error_message}
                </pre>
                <p>Please check the logs for more details.</p>
            </body>
        </html>
        """

        text_content = f"Pipeline Error: {error_message}"

        # Send to admin emails only
        admin_emails = os.getenv("ADMIN_EMAILS", "").split(",")
        tasks = []

        for email in admin_emails:
            if email:
                tasks.append(self._send_email(email.strip(), subject, html_content, text_content))

        if self.slack_webhook:
            tasks.append(
                self._send_slack_message(
                    {
                        "text": subject,
                        "attachments": [
                            {
                                "color": "danger",
                                "fields": [
                                    {"title": "Error", "value": error_message, "short": False}
                                ],
                            }
                        ],
                    }
                )
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _create_opportunity_html(self, opportunities: pd.DataFrame) -> str:
        """Create HTML content for opportunity notifications"""
        top_5 = opportunities.head(5)

        rows_html = ""
        for _, opp in top_5.iterrows():
            rows_html += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">{opp['org1_account_name']}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{opp['org2_account_name']}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{opp['score']:.2f}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">${opp['estimated_value']:,.0f}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{opp['next_best_action']}</td>
            </tr>
            """

        total_value = opportunities["estimated_value"].sum()

        return f"""
        <html>
            <body style="font-family: Arial, sans-serif;">
                <h2 style="color: #1976d2;">New Cross-Sell Opportunities Identified</h2>

                <p>We've identified <strong>{len(opportunities)}</strong> new high-value cross-sell
                opportunities with a total potential value of <strong>${total_value:,.0f}</strong>.</p>

                <h3>Top 5 Opportunities:</h3>
                <table style="border-collapse: collapse; width: 100%;">
                    <thead>
                        <tr style="background-color: #f0f0f0;">
                            <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Account 1</th>
                            <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Account 2</th>
                            <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Score</th>
                            <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Est. Value</th>
                            <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Next Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>

                <p style="margin-top: 20px;">
                    <a href="{os.getenv('DASHBOARD_URL', 'http://localhost:8501')}"
                       style="background-color: #1976d2; color: white; padding: 10px 20px;
                              text-decoration: none; border-radius: 5px; display: inline-block;">
                        View All Opportunities
                    </a>
                </p>
            </body>
        </html>
        """

    def _create_opportunity_text(self, opportunities: pd.DataFrame) -> str:
        """Create plain text content for opportunity notifications"""
        top_5 = opportunities.head(5)
        total_value = opportunities["estimated_value"].sum()

        text = f"""New Cross-Sell Opportunities Identified

We've identified {len(opportunities)} new high-value cross-sell opportunities
with a total potential value of ${total_value:,.0f}.

Top 5 Opportunities:
"""

        for i, (_, opp) in enumerate(top_5.iterrows(), 1):
            text += f"""
{i}. {opp['org1_account_name']} × {opp['org2_account_name']}
   Score: {opp['score']:.2f} | Value: ${opp['estimated_value']:,.0f}
   Action: {opp['next_best_action']}
"""

        text += f"\nView all opportunities: {os.getenv('DASHBOARD_URL', 'http://localhost:8501')}"

        return text

    async def _send_email(self, recipient: str, subject: str, html_content: str, text_content: str):
        """Send email notification"""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.smtp_config["from_email"]
            msg["To"] = recipient

            text_part = MIMEText(text_content, "plain")
            html_part = MIMEText(html_content, "html")

            msg.attach(text_part)
            msg.attach(html_part)

            # Send email in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_email_sync, recipient, msg)

            logger.info(f"Email sent to {recipient}")

        except Exception as e:
            logger.error(f"Failed to send email to {recipient}: {e}")

    def _send_email_sync(self, recipient: str, msg: MIMEMultipart):
        """Synchronous email sending"""
        with smtplib.SMTP(self.smtp_config["host"], self.smtp_config["port"]) as server:
            server.starttls()
            server.login(self.smtp_config["username"], self.smtp_config["password"])
            server.send_message(msg)

    async def _send_slack(self, subject: str, opportunities: pd.DataFrame):
        """Send Slack notification"""
        if not self.slack_webhook:
            return

        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": subject}},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"Total opportunities: *{len(opportunities)}*\n"
                    f"Total value: *${opportunities['estimated_value'].sum():,.0f}*",
                },
            },
        ]

        # Add top opportunities
        for _, opp in opportunities.head(3).iterrows():
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{opp['org1_account_name']}* × *{opp['org2_account_name']}*\n"
                        f"Score: {opp['score']:.2f} | Value: ${opp['estimated_value']:,.0f}",
                    },
                    "accessory": {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Details"},
                        "url": f"{os.getenv('DASHBOARD_URL', 'http://localhost:8501')}",
                    },
                }
            )

        await self._send_slack_message({"blocks": blocks})

    async def _send_slack_message(self, payload: Dict[str, Any]):
        """Send message to Slack webhook"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.slack_webhook, json=payload) as response:
                    if response.status == 200:
                        logger.info("Slack notification sent")
                    else:
                        logger.error(f"Slack notification failed: {response.status}")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

    async def _send_teams(self, subject: str, opportunities: pd.DataFrame):
        """Send Microsoft Teams notification"""
        if not self.teams_webhook:
            return

        # Create Teams adaptive card
        card = {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "summary": subject,
            "themeColor": "0078D7",
            "sections": [
                {
                    "activityTitle": subject,
                    "facts": [
                        {"name": "Total Opportunities", "value": str(len(opportunities))},
                        {
                            "name": "Total Value",
                            "value": f"${opportunities['estimated_value'].sum():,.0f}",
                        },
                    ],
                }
            ],
            "potentialAction": [
                {
                    "@type": "OpenUri",
                    "name": "View Dashboard",
                    "targets": [
                        {
                            "os": "default",
                            "uri": os.getenv("DASHBOARD_URL", "http://localhost:8501"),
                        }
                    ],
                }
            ],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.teams_webhook, json=card) as response:
                    if response.status == 200:
                        logger.info("Teams notification sent")
                    else:
                        logger.error(f"Teams notification failed: {response.status}")
        except Exception as e:
            logger.error(f"Failed to send Teams notification: {e}")
