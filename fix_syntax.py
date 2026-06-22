import re

with open("chart_app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix the single brackets to double brackets for f-string formatting
bad_css = """/* Fix Light Gray Areas */
[data-testid="stHeader"] {
    background-color: transparent !important;
}
.stApp > header {
    background-color: transparent !important;
}
[data-testid="stToolbar"] {
    right: 2rem;
}
/* Ensure the main background completely covers everything */
.stApp, .main {
    background-color: var(--paper) !important;
}"""

good_css = """/* Fix Light Gray Areas */
[data-testid="stHeader"] {{
    background-color: transparent !important;
}}
.stApp > header {{
    background-color: transparent !important;
}}
[data-testid="stToolbar"] {{
    right: 2rem;
}}
/* Ensure the main background completely covers everything */
.stApp, .main {{
    background-color: var(--paper) !important;
}}"""

if bad_css in content:
    content = content.replace(bad_css, good_css)

with open("chart_app.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed syntax error")
