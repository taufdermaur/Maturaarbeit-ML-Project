"""
EPS to PNG Converter with GUI Interface
Built for Visual Studio Code

This script provides a simple GUI for converting EPS files to PNG format.
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path


class EPSToPNGConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("EPS to PNG Converter")
        self.root.geometry("600x450")
        self.root.resizable(True, True)
        
        # Set app icon if available
        try:
            self.root.iconbitmap("icon.ico")  # You can add your own icon file
        except:
            pass
            
        # Variables
        self.file_paths = []
        self.output_dir = ""
        self.resolution = tk.StringVar(value="300")
        
        # Create the UI
        self.create_widgets()
        
        # Check Ghostscript installation
        self.check_ghostscript()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Select EPS Files", padding="10")
        file_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        select_btns_frame = ttk.Frame(file_frame)
        select_btns_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(select_btns_frame, text="Select Files", command=self.select_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_btns_frame, text="Select Folder", command=self.select_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_btns_frame, text="Clear Selection", command=self.clear_selection).pack(side=tk.LEFT, padx=5)
        
        # Files list
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Scrollbars
        scrollbar_y = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        scrollbar_x = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL)
        
        # Files listbox
        self.files_listbox = tk.Listbox(
            list_frame, 
            selectmode=tk.EXTENDED,
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set
        )
        
        scrollbar_y.config(command=self.files_listbox.yview)
        scrollbar_x.config(command=self.files_listbox.xview)
        
        # Layout
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # File counter
        self.file_counter = ttk.Label(file_frame, text="0 files selected")
        self.file_counter.pack(anchor=tk.W, pady=5)
        
        # Output directory section
        output_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding="10")
        output_frame.pack(fill=tk.X, pady=5)
        
        # Output directory
        out_dir_frame = ttk.Frame(output_frame)
        out_dir_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(out_dir_frame, text="Output Directory:").pack(side=tk.LEFT, padx=5)
        self.output_dir_var = tk.StringVar(value="Same as input files")
        self.output_dir_label = ttk.Label(out_dir_frame, textvariable=self.output_dir_var, width=40, 
                                    background="#f0f0f0", padding=5)
        self.output_dir_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(out_dir_frame, text="Browse...", command=self.select_output_dir).pack(side=tk.LEFT, padx=5)
        
        # Resolution
        res_frame = ttk.Frame(output_frame)
        res_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(res_frame, text="Resolution (DPI):").pack(side=tk.LEFT, padx=5)
        resolution_entry = ttk.Entry(res_frame, textvariable=self.resolution, width=5)
        resolution_entry.pack(side=tk.LEFT, padx=5)
        
        # Convert button
        convert_frame = ttk.Frame(main_frame)
        convert_frame.pack(fill=tk.X, pady=10)
        
        self.progress = ttk.Progressbar(convert_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, pady=10)
        
        self.convert_btn = ttk.Button(convert_frame, text="Convert", command=self.convert_files)
        self.convert_btn.pack(pady=5, padx=50, fill=tk.X)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, border=1, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def check_ghostscript(self):
        """Check if Ghostscript is installed"""
        try:
            subprocess.run(['gs', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            self.status_var.set("Ready - Ghostscript found")
        except (subprocess.SubprocessError, FileNotFoundError):
            self.status_var.set("Warning: Ghostscript not found. Please install it first.")
            messagebox.showwarning(
                "Ghostscript Not Found", 
                "Ghostscript is required but not found. Please install it:\n\n"
                "- Windows: Download from ghostscript.com\n"
                "- Linux: sudo apt-get install ghostscript\n"
                "- macOS: brew install ghostscript"
            )
    
    def select_files(self):
        """Open file dialog to select EPS files"""
        files = filedialog.askopenfilenames(
            title="Select EPS Files",
            filetypes=[("EPS files", "*.eps"), ("All files", "*.*")]
        )
        
        if files:
            # Add to existing list
            self.file_paths.extend(files)
            # Remove duplicates
            self.file_paths = list(dict.fromkeys(self.file_paths))
            # Update listbox
            self.update_files_list()
    
    def select_folder(self):
        """Open folder dialog to select a directory with EPS files"""
        folder = filedialog.askdirectory(title="Select Folder Containing EPS Files")
        
        if folder:
            # Find all EPS files in the directory
            eps_files = []
            for file in os.listdir(folder):
                if file.lower().endswith('.eps'):
                    eps_files.append(os.path.join(folder, file))
            
            if eps_files:
                # Add to existing list
                self.file_paths.extend(eps_files)
                # Remove duplicates
                self.file_paths = list(dict.fromkeys(self.file_paths))
                # Update listbox
                self.update_files_list()
                messagebox.showinfo("Files Found", f"Found {len(eps_files)} EPS files in the folder.")
            else:
                messagebox.showinfo("No Files Found", "No EPS files found in the selected folder.")
    
    def clear_selection(self):
        """Clear the list of selected files"""
        self.file_paths = []
        self.update_files_list()
    
    def update_files_list(self):
        """Update the listbox with current file paths"""
        self.files_listbox.delete(0, tk.END)
        for file in self.file_paths:
            self.files_listbox.insert(tk.END, file)
        
        # Update counter
        count = len(self.file_paths)
        self.file_counter.config(text=f"{count} files selected")
    
    def select_output_dir(self):
        """Select output directory"""
        folder = filedialog.askdirectory(title="Select Output Directory")
        
        if folder:
            self.output_dir = folder
            # Truncate path if too long
            display_path = folder
            if len(display_path) > 40:
                display_path = "..." + display_path[-37:]
            self.output_dir_var.set(display_path)
        else:
            self.output_dir = ""
            self.output_dir_var.set("Same as input files")
    
    def convert_files(self):
        """Convert all selected EPS files to PNG"""
        if not self.file_paths:
            messagebox.showinfo("No Files", "Please select at least one EPS file to convert.")
            return
        
        try:
            resolution = int(self.resolution.get())
            if resolution <= 0:
                raise ValueError("Resolution must be a positive number")
        except ValueError:
            messagebox.showerror("Invalid Resolution", "Please enter a valid positive number for resolution.")
            return
        
        # Disable buttons during conversion
        self.convert_btn.config(state=tk.DISABLED)
        
        # Set up progress bar
        total_files = len(self.file_paths)
        self.progress["maximum"] = total_files
        self.progress["value"] = 0
        
        success_count = 0
        fail_count = 0
        
        for index, eps_file in enumerate(self.file_paths):
            # Update status
            file_name = os.path.basename(eps_file)
            self.status_var.set(f"Converting {index+1}/{total_files}: {file_name}")
            self.root.update()
            
            # Determine output file
            if self.output_dir:
                output_file = os.path.join(self.output_dir, os.path.basename(eps_file).replace('.eps', '.png'))
            else:
                output_file = str(Path(eps_file).with_suffix('.png'))
            
            # Convert file
            if self.convert_eps_to_png(eps_file, output_file, resolution):
                success_count += 1
            else:
                fail_count += 1
            
            # Update progress
            self.progress["value"] = index + 1
            self.root.update()
        
        # Conversion complete
        self.status_var.set(f"Conversion complete: {success_count} successful, {fail_count} failed")
        self.convert_btn.config(state=tk.NORMAL)
        
        # Show summary
        messagebox.showinfo(
            "Conversion Complete", 
            f"Conversion complete:\n\n"
            f"- {success_count} files converted successfully\n"
            f"- {fail_count} files failed"
        )

    def convert_eps_to_png(self, eps_file, output_file, resolution):
        """Convert a single EPS file to PNG using Ghostscript"""
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Ghostscript command
            GHOSTSCRIPT_COMMAND = r"C:\Program Files\gs\gs10.05.1\bin\gswin64c.exe"

            command = [
                GHOSTSCRIPT_COMMAND,
                '-dSAFER',
                '-dBATCH',
                '-dNOPAUSE',
                f'-r{resolution}',
                '-sDEVICE=pngalpha',
                f'-sOutputFile={output_file}',
                eps_file
            ]
            
            # Execute command
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        
        except Exception as e:
            print(f"Error converting {eps_file}: {e}")
            return False


def main():
    root = tk.Tk()
    app = EPSToPNGConverter(root)
    root.mainloop()


if __name__ == "__main__":
    main()
