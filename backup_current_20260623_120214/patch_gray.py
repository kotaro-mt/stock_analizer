import re

with open("chart_app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add CSS rules to hide/transparentize the light gray header and other potential light gray areas.
css_addition = """
/* Fix Light Gray Areas */
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
}
"""

if "/* Fix Light Gray Areas */" not in content:
    content = content.replace("</style>", css_addition + "\n</style>")

with open("chart_app.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Patched light gray areas in chart_app.py")
