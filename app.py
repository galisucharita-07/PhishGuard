
from flask import Flask, render_template, request
import joblib
import socket
import ipaddress
from urllib.parse import urlparse

app = Flask(__name__)

# Load the trained phishing detection model
model = joblib.load("phishing_model.joblib")


# Trusted website root domains
TRUSTED_DOMAINS = {
    # Search, email and productivity
    "google.com",
    "google.co.in",
    "gmail.com",
    "youtube.com",
    "bing.com",
    "yahoo.com",
    "outlook.com",
    "office.com",
    "microsoft365.com",

    # AI platforms
    "openai.com",
    "chatgpt.com",
    "claude.ai",
    "anthropic.com",
    "huggingface.co",
    "perplexity.ai",

    # Coding and developer platforms
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "stackoverflow.com",
    "stackexchange.com",
    "leetcode.com",
    "geeksforgeeks.org",
    "hackerrank.com",
    "codechef.com",
    "codeforces.com",
    "replit.com",
    "codesandbox.io",
    "npmjs.com",
    "pypi.org",
    "python.org",
    "nodejs.org",
    "react.dev",
    "developer.mozilla.org",

    # Data science and machine learning
    "kaggle.com",
    "jupyter.org",
    "pytorch.org",
    "tensorflow.org",
    "scikit-learn.org",

    # Education and online learning
    "coursera.org",
    "edx.org",
    "udemy.com",
    "freecodecamp.org",
    "w3schools.com",
    "infosys.com",
    "swayam.gov.in",
    "nptel.ac.in",
    "mit.edu",
    "harvard.edu",
    "stanford.edu",

    # Cloud and technology
    "amazon.com",
    "amazon.in",
    "aws.amazon.com",
    "cloud.google.com",
    "azure.microsoft.com",
    "cloudflare.com",
    "vercel.com",
    "netlify.com",
    "render.com",
    "mongodb.com",
    "firebase.google.com",
    "supabase.com",
    "docker.com",

    # Professional and career platforms
    "linkedin.com",
    "naukri.com",
    "wellfound.com",
    "internshala.com",
    "indeed.com",
    "glassdoor.com",
    "forage.com",
    "smartrecruiters.com",
    "greenhouse.io",
    "lever.co",

    # Government and public services
    "india.gov.in",
    "uidai.gov.in",
    "incometax.gov.in",
    "digilocker.gov.in",
    "isro.gov.in",
    "nic.in",
    "meity.gov.in",

    # Reference and documentation
    "wikipedia.org",
    "britannica.com",
    "archive.org",

    # Social and communication platforms
    "reddit.com",
    "instagram.com",
    "facebook.com",
    "whatsapp.com",
    "telegram.org",
    "discord.com",
    "slack.com",
    "zoom.us",

    # Shopping, payments and travel
    "flipkart.com",
    "myntra.com",
    "ajio.com",
    "paypal.com",
    "stripe.com",
    "booking.com",
    "makemytrip.com",

    # Design and content creation
    "canva.com",
    "figma.com",
    "adobe.com",
    "medium.com",
    "wordpress.com",
    "notion.so",

    # Entertainment
    "netflix.com",
    "spotify.com",
    "twitch.tv",
    "disneyplus.com",
}


def normalize_url(raw_url):
    """Validate and normalize the submitted URL."""
    raw_url = raw_url.strip()

    if not raw_url or len(raw_url) > 2048:
        return None, None

    # Add a scheme when the user enters only a domain or URL
    if "://" not in raw_url:
        raw_url = "https://" + raw_url

    try:
        parsed = urlparse(raw_url)

        if parsed.scheme not in ("http", "https"):
            return None, None

        hostname = parsed.hostname

        if not hostname or "." not in hostname:
            return None, None

        # Validate the port if one is provided
        _ = parsed.port

        return raw_url, parsed

    except (ValueError, UnicodeError):
        return None, None


def check_domain(hostname):
    """Check whether the hostname resolves through DNS."""
    try:
        ipaddress.ip_address(hostname)
        return "reachable"
    except ValueError:
        pass

    try:
        socket.getaddrinfo(hostname, None)
        return "reachable"
    except socket.gaierror:
        return "not_found"
    except OSError:
        return "unknown"


def is_trusted_domain(hostname):
    """Match trusted domains and their subdomains safely."""
    return any(
        hostname == domain or hostname.endswith("." + domain)
        for domain in TRUSTED_DOMAINS
    )


def analyze_url(raw_url):
    """Analyze a URL using the trusted list and ML model."""
    normalized_url, parsed = normalize_url(raw_url)

    if not normalized_url or not parsed:
        return {
            "status": "invalid",
            "submitted_url": raw_url,
            "score": 0,
            "level": "Unknown",
            "findings": [
                "Please enter a valid HTTP or HTTPS URL."
            ]
        }

    hostname = (parsed.hostname or "").lower().rstrip(".")

    # Reject malformed hostnames that could otherwise bypass checks
    if not hostname:
        return {
            "status": "invalid",
            "submitted_url": raw_url,
            "score": 0,
            "level": "Unknown",
            "findings": ["The hostname is invalid."]
        }

    # Trusted-domain exception comes before DNS and ML checks
    if is_trusted_domain(hostname):
        return {
            "status": "analyzed",
            "submitted_url": raw_url,
            "score": 0.0,
            "level": "No Risk",
            "findings": [
                "The hostname matches your configured trusted-domain list.",
                "The allowlist assigns a phishing score of 0.00%.",
                "This is a configured exception, not a guarantee of safety."
            ]
        }

    # Check DNS for other domains
    domain_status = check_domain(hostname)

    if domain_status == "not_found":
        return {
            "status": "not_found",
            "submitted_url": raw_url,
            "score": 0,
            "level": "Unknown",
            "findings": [
                "The hostname could not be resolved through DNS.",
                "Check the URL spelling or try again later.",
                "A DNS failure alone does not prove a URL is phishing."
            ]
        }

    # Use the trained ML model for non-allowlisted URLs
    try:
        probabilities = model.predict_proba([normalized_url])[0]
        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError("Phishing class is missing from the model.")

        phishing_index = classes.index(1)
        score = float(probabilities[phishing_index]) * 100
        prediction = int(model.predict([normalized_url])[0])

    except (ValueError, AttributeError, IndexError) as error:
        app.logger.exception("Model prediction failed: %s", error)

        return {
            "status": "error",
            "submitted_url": raw_url,
            "score": 0,
            "level": "Unknown",
            "findings": [
                "The model could not analyze this URL.",
                "Check that phishing_model.joblib is the expected trained model."
            ]
        }

    findings = [
        "The URL was evaluated using character-level URL patterns.",
        "The model score is an estimate, not a calibrated probability.",
        "The model result alone cannot confirm whether a website is safe."
    ]

    if prediction == 1:
        if score >= 70:
            level = "High"
        elif score >= 40:
            level = "Medium"
        else:
            level = "Low"

        findings.append(
            "The model detected URL patterns commonly associated with phishing."
        )
    else:
        level = "Low"
        findings.append(
            "The model did not classify this URL as phishing, but caution is still advised."
        )

    if domain_status == "unknown":
        findings.append(
            "The DNS check could not determine the hostname's status."
        )

    return {
        "status": "analyzed",
        "submitted_url": raw_url,
        "score": round(score, 2),
        "level": level,
        "findings": findings
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    submitted_url = ""

    if request.method == "POST":
        submitted_url = request.form.get("url", "").strip()

        if len(submitted_url) > 2048:
            result = {
                "status": "invalid",
                "submitted_url": submitted_url[:2048],
                "score": 0,
                "level": "Unknown",
                "findings": [
                    "The URL must be 2048 characters or fewer."
                ]
            }
        else:
            result = analyze_url(submitted_url)

    return render_template(
        "index.html",
        result=result,
        submitted_url=submitted_url
    )


if __name__ == "__main__":
    app.run(debug=True)
