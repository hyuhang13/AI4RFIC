import numpy as np
import os
from datetime import datetime

def generate_binary_matrices(num_matrices=100000, matrix_size=19, output_file="binary_matrices.txt"):
    
    
    start_time = datetime.now()
    
    # È·±£Êä³öÄ¿Â¼´æÔÚ
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Ê¹ÓÃÁ÷Ê½Ð´Èë£¬±ÜÃâÄÚ´æÕ¼ÓÃ¹ý´ó
    with open(output_file, 'w') as f:
        for matrix_idx in range(1, num_matrices + 1):
            # Éú³ÉËæ»ú¶þ½øÖÆ¾ØÕó (0»ò1)
            matrix = np.random.randint(0, 2, size=(matrix_size, matrix_size))
            
            # Ð´Èë¾ØÕó£¬Ã¿ÐÐ×÷ÎªÒ»ÐÐÎÄ±¾
            for row in range(matrix_size):
                # ½«Ã¿ÐÐ×ª»»Îª×Ö·û´® (0ºÍ1Ö®¼äÃ»ÓÐ¿Õ¸ñ)
                row_str = ','.join(str(matrix[row, col]) for col in range(matrix_size))
                f.write(row_str + '\n')
            
            # ÔÚ¾ØÕóÖ®¼äÌí¼ÓÒ»¸ö¿ÕÐÐ×÷Îª·Ö¸ô·û
            #f.write()
            f.write(f'# Matrix Index: {matrix_idx}\n')
            # ÏÔÊ¾½ø¶È
            if matrix_idx % 10000 == 0:
                progress = (matrix_idx / num_matrices) * 100
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"{progress:.1f}% ({matrix_idx}/{num_matrices})")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    file_size = os.path.getsize(output_file)
    
    
    print(f" {elapsed:.1f} ")
    print(f"{num_matrices/elapsed:.1f} ")
    print(f"{file_size / (1024*1024):.2f} MB")
    print(f"{os.path.abspath(output_file)}")

# ÔËÐÐÉú³Éº¯Êý
if __name__ == "__main__":
    # ÅäÖÃ²ÎÊý - ¿É¸ù¾ÝÐèÒªÐÞ¸Ä
    NUM_MATRICES = 100000
    MATRIX_SIZE = 19
    OUTPUT_FILE = "binary_matrices.txt"
    
    # Éú³É¾ØÕóÎÄ¼þ
    generate_binary_matrices(
        num_matrices=NUM_MATRICES,
        matrix_size=MATRIX_SIZE,
        output_file=OUTPUT_FILE
    )