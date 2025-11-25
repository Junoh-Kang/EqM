import os
import numpy as np
from PIL import Image
import json

def get_diff(p1, p2):
    if not os.path.exists(p1) or not os.path.exists(p2):
        return -1.0
    try:
        im1 = np.array(Image.open(p1)).astype(float)
        im2 = np.array(Image.open(p2)).astype(float)
        return np.mean((im1 - im2)**2)
    except Exception as e:
        print(f"Error reading {p1} or {p2}: {e}")
        return -1.0

def main():
    base_dir = "output/grad_tests"
    out_dir = os.path.join(base_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    
    samples = []
    print("Calculating differences for 1000 samples...")
    
    for i in range(1000):
        sample_id = f"{i:06d}"
        
        # Paths relative to the script execution (root)
        p1_250 = os.path.join(base_dir, "one_class_500/step_250", sample_id + ".png")
        p1_500 = os.path.join(base_dir, "one_class_500/step_500", sample_id + ".png")
        
        p2_250 = os.path.join(base_dir, "two_class_500/step_250", sample_id + ".png")
        p2_500 = os.path.join(base_dir, "two_class_500/step_500", sample_id + ".png")
        
        diff_one = get_diff(p1_250, p1_500)
        diff_two = get_diff(p2_250, p2_500)
        
        samples.append({
            "id": i,
            "str_id": sample_id,
            "diff_one": diff_one,
            "diff_two": diff_two
        })
        
        if i % 100 == 0:
            print(f"Processed {i} samples...")

    # Sort by diff_one descending initially
    samples.sort(key=lambda x: x["diff_one"], reverse=True)
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Grad Test Comparison (1000 Samples)</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
        th, td {{ border: 1px solid #ddd; padding: 4px; text-align: center; font-size: 12px; }}
        th {{ background-color: #f2f2f2; position: sticky; top: 0; z-index: 10; }}
        img {{ width: 100%; max-width: 150px; height: auto; display: block; margin: 0 auto; }}
        .one-class {{ background-color: #e6f7ff; }}
        .two-class {{ background-color: #fff0f0; }}
        .controls {{ margin-bottom: 20px; padding: 10px; background: #eee; position: sticky; top: 0; z-index: 20; }}
        .sample-group {{ border-bottom: 2px solid #999; }}
    </style>
    <script>
        var samples = {json.dumps(samples)};
        
        function renderTable(sortKey) {{
            var tbody = document.getElementById('table-body');
            tbody.innerHTML = '';
            
            // Sort data
            samples.sort(function(a, b) {{
                return b[sortKey] - a[sortKey]; // Descending
            }});
            
            // Render
            // Use a document fragment for performance
            var fragment = document.createDocumentFragment();
            
            samples.forEach(function(s) {{
                var sid = s.str_id;
                
                // One Class Row
                var tr1 = document.createElement('tr');
                tr1.className = 'one-class';
                tr1.innerHTML = `
                    <td><strong>one_class</strong><br>ID: ${{s.id}}<br>Diff: ${{s.diff_one.toFixed(2)}}</td>
                    <td><img loading="lazy" src="../one_class_500/step_100/${{sid}}.png"></td>
                    <td><img loading="lazy" src="../one_class_500/step_200/${{sid}}.png"></td>
                    <td><img loading="lazy" src="../one_class_500/step_250/${{sid}}.png"></td>
                    <td><img loading="lazy" src="../one_class_500/step_300/${{sid}}.png"></td>
                    <td><img loading="lazy" src="../one_class_500/step_400/${{sid}}.png"></td>
                    <td><img loading="lazy" src="../one_class_500/step_500/${{sid}}.png"></td>
                `;
                fragment.appendChild(tr1);
                
                // Two Class Row
                var tr2 = document.createElement('tr');
                tr2.className = 'two-class sample-group';
                tr2.innerHTML = `
                    <td><strong>two_class</strong><br>ID: ${{s.id}}<br>Diff: ${{s.diff_two.toFixed(2)}}</td>
                    <td><img loading="lazy" src="../two_class_500/step_100/${{sid}}.png"></td>
                    <td><img loading="lazy" src="../two_class_500/step_200/${{sid}}.png"></td>
                    <td><img loading="lazy" src="../two_class_500/step_250/${{sid}}.png"></td>
                    <td><img loading="lazy" src="../two_class_500/step_300/${{sid}}.png"></td>
                    <td><img loading="lazy" src="../two_class_500/step_400/${{sid}}.png"></td>
                    <td><img loading="lazy" src="../two_class_500/step_500/${{sid}}.png"></td>
                `;
                fragment.appendChild(tr2);
            }});
            
            tbody.appendChild(fragment);
        }}
        
        window.onload = function() {{
            renderTable('diff_one');
        }};
    </script>
</head>
<body>
    <div class="controls">
        <label>Sort by Difference (Step 250 vs 500): </label>
        <button onclick="renderTable('diff_one')">Sort by One Class Diff</button>
        <button onclick="renderTable('diff_two')">Sort by Two Class Diff</button>
    </div>

    <table>
        <thead>
            <tr>
                <th style="width: 100px;">Info</th>
                <th>Step 100</th>
                <th>Step 200</th>
                <th>Step 250</th>
                <th>Step 300</th>
                <th>Step 400</th>
                <th>Step 500</th>
            </tr>
        </thead>
        <tbody id="table-body">
            <!-- Content rendered by JS -->
        </tbody>
    </table>
</body>
</html>'''

    with open(os.path.join(out_dir, 'index.html'), 'w') as f:
        f.write(html_content)
    print(f"Generated {os.path.join(out_dir, 'index.html')}")

if __name__ == "__main__":
    main()
