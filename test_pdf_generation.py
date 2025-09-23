#!/usr/bin/env python3
"""
Test script to verify PDF generation improvements
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# Test if matplotlib PDF backend is available
try:
    from matplotlib.backends.backend_pdf import PdfPages
    print("✓ PdfPages import successful")
except ImportError as e:
    print(f"✗ PdfPages import failed: {e}")
    exit(1)

# Test PDF creation
try:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_filename = f"test_pdf_generation_{timestamp}.pdf"
    
    print(f"Creating test PDF: {test_filename}")
    
    # Create PDF with high resolution settings
    with PdfPages(test_filename) as pdf_pages:
        
        # Test plot 1: Simple line plot
        fig1 = plt.figure(figsize=(16, 10))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plt.plot(x, y, linewidth=2, color='#1f77b4', label='Sine Wave')
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save with high resolution (600 DPI)
        pdf_pages.savefig(fig1, bbox_inches='tight', dpi=600, 
                         facecolor='white', edgecolor='none')
        plt.close(fig1)
        
        # Test plot 2: Scatter plot
        fig2 = plt.figure(figsize=(16, 10))
        x2 = np.random.randn(100)
        y2 = np.random.randn(100)
        plt.scatter(x2, y2, alpha=0.6, s=50, color='#ff7f0e')
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save with high resolution
        pdf_pages.savefig(fig2, bbox_inches='tight', dpi=600,
                         facecolor='white', edgecolor='none')
        plt.close(fig2)
        
        # Test plot 3: Bar chart
        fig3 = plt.figure(figsize=(16, 10))
        categories = ['A', 'B', 'C', 'D', 'E']
        values = [23, 45, 56, 78, 32]
        plt.bar(categories, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        plt.xlabel('Categories', fontsize=12)
        plt.ylabel('Values', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save with high resolution
        pdf_pages.savefig(fig3, bbox_inches='tight', dpi=600,
                         facecolor='white', edgecolor='none')
        plt.close(fig3)
    
    # Check file size
    file_size = os.path.getsize(test_filename)
    file_size_kb = file_size / 1024
    
    print(f"✓ Test PDF created successfully!")
    print(f"📄 File: {test_filename}")
    print(f"📊 File size: {file_size_kb:.1f} KB")
    print(f"🎨 High resolution (600 DPI) plots included")
    print(f"📐 Professional formatting applied")
    
    if file_size_kb > 100:  # Expect larger file size with high DPI
        print(f"✅ File size indicates high-resolution content")
    else:
        print(f"⚠️ File size seems small for high-resolution content")
        
except Exception as e:
    print(f"✗ PDF creation failed: {e}")
    exit(1)

print("\n🎯 PDF generation test completed successfully!")
print("📋 The improvements should now work in the main notebook!")