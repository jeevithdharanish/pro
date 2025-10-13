import { Copy } from 'lucide-react';
import { useState } from 'react';

export default function CodeGallery() {
  const codes = [
    {
      title: "1️ Date / Time / User Info",
      filename: "date_time_user.sh",
      code: `#!/bin/bash
echo "Date: \$(date +'%Y-%m-%d')"
echo "Time: \$(date +'%H:%M:%S')"
echo "Username: \$USER"
echo "Current Directory: \$(pwd)"`
    },
    {
      title: "2️ Sum of Digits of a Number",
      filename: "sum_digits.sh",
      code: `#!/bin/bash
echo "Enter a number:"
read num
sum=0
for (( i=0; i<\${#num}; i++ ))
do
  digit=\${num:i:1}
  sum=\$((sum + digit))
done
echo "Sum of digits = \$sum"`
    },
    {
      title: "3️ Student Total & Average",
      filename: "student_average.sh",
      code: `#!/bin/bash
echo "Enter Student Name:"
read name
echo "Enter marks for 5 subjects (space-separated):"
read m1 m2 m3 m4 m5
total=\$((m1 + m2 + m3 + m4 + m5))
average=\$((total / 5))
echo "Student Name: \$name"
echo "Total Marks: \$total"
echo "Average Marks: \$average"`
    },
    {
      title: "4️ Prime Number Check",
      filename: "prime_check.sh",
      code: `#!/bin/bash
read -p "Enter a number: " num
if (( num <= 1 )); then
  echo "\$num is not prime."
  exit 0
fi
for (( i=2; i*i<=num; i++ )); do
  if (( num % i == 0 )); then
    echo "\$num is not prime."
    exit 0
  fi
done
echo "\$num is prime."`
    },
    {
      title: "5️ Factorial",
      filename: "factorial.sh",
      code: `#!/bin/bash
read -p "Enter a number: " num
fact=1
for (( i=2; i<=num; i++ )); do
  fact=\$((fact * i))
done
echo "Factorial of \$num is \$fact"`
    },
    {
      title: "6️ Palindrome Number",
      filename: "palindrome_number.sh",
      code: `#!/bin/bash
read -p "Enter a number: " num
rev=0
temp=\$num
while (( temp > 0 )); do
  digit=\$(( temp % 10 ))
  rev=\$(( rev * 10 + digit ))
  temp=\$(( temp / 10 ))
done
if (( num == rev )); then
  echo "\$num is a palindrome."
else
  echo "\$num is not a palindrome."
fi`
    },
    {
      title: "7️ Reverse a String (from arg)",
      filename: "reverse_string.sh",
      code: `#!/bin/bash
if [ \$# -eq 0 ]; then
  echo "Usage: \$0 string"
  exit 1
fi
input="\$1"
len=\${#input}
rev=""
for (( i=len-1; i>=0; i-- )); do
  rev="\$rev\${input:\$i:1}"
done
echo "Reversed string: \$rev"`
    },
    {
      title: "8️ Search Element in Array",
      filename: "search_array.sh",
      code: `#!/bin/bash
arr=(10 20 30 40 50)
read -p "Enter element to search: " ele
found=0
for i in "\${arr[@]}"; do
  if [ "\$i" -eq "\$ele" ]; then
    found=1
    break
  fi
done
if (( found == 1 )); then
  echo "\$ele found in array."
else
  echo "\$ele not found in array."
fi`
    },
    {
      title: "9️ Gross Salary Calculation",
      filename: "gross_salary.sh",
      code: `#!/bin/bash
read -p "Enter basic salary: " basic

if (( basic < 1500 )); then
  hra=\$(( basic * 10 / 100 ))
  da=\$(( basic * 90 / 100 ))
else
  hra=500
  da=\$(( basic * 98 / 100 ))
fi

gross=\$(( basic + hra + da ))
echo "Gross salary = \$gross"`
    },
    {
      title: "10️ File Line & Word Count",
      filename: "file_stats.sh",
      code: `#!/bin/bash
read -p "Enter filename: " file

if [ ! -f "\$file" ]; then
  echo "File not found."
  exit 1
fi

lines=\$(wc -l < "\$file")
words=\$(wc -w < "\$file")

echo "Number of lines: \$lines"
echo "Number of words: \$words"`
    },
    {
      title: "11️ Delete Lines Containing a Word (with backup)",
      filename: "delete_lines.sh",
      code: `#!/bin/bash
if [ \$# -lt 2 ]; then
  echo "Usage: \$0 word file1 [file2 ...]"
  exit 1
fi

word=\$1
shift

for file in "\$@"; do
  if [ ! -f "\$file" ]; then
    echo "File \$file not found."
    continue
  fi
  # Create backup before modifying
  cp "\$file" "\${file}.bak"
  sed -i "/\$word/d" "\$file"
  echo "Deleted lines containing '\$word' from \$file"
done`
    },
    {
      title: "12️ String Case Conversions",
      filename: "case_conversion.sh",
      code: `#!/bin/bash
echo "Enter a string:"
read str

echo "Uppercase: \${str^^}"
echo "Lowercase: \${str,,}"
echo "Title Case: \$(echo "\$str" | sed -e 's/\\b\\(.\)/\\u\\1/g')"`
    },
    {
      title: "13️ Merge & Sort Two Files",
      filename: "merge_sort_files.sh",
      code: `#!/bin/bash
echo "Enter first file:"
read f1
echo "Enter second file:"
read f2
echo "Enter output file:"
read f3

cat "\$f1" "\$f2" | sort > "\$f3"
echo "Merged and sorted content stored in \$f3"`
    },
    {
      title: "14️ Simple Menu Script",
      filename: "menu.sh",
      code: `#!/bin/bash
while true; do
    echo "----- MENU -----"
    echo "1. Display Date"
    echo "2. Display Current Directory"
    echo "3. List Files"
    echo "4. Exit"
    echo "Enter your choice: "
    read ch

    case \$ch in
        1) date ;;
        2) pwd ;;
        3) ls -l ;;
        4) echo "Exiting..."; break ;;
        *) echo "Invalid choice!" ;;
    esac
    echo ""
done`
    },
    {
      title: "15️ Change Permissions Prompt",
      filename: "chmod_prompt.sh",
      code: `#!/bin/bash
echo "Enter file/directory name:"
read target

if [ -e "\$target" ]; then
    echo "Enter permission (e.g., 755 or 644):"
    read perm
    chmod \$perm "\$target"
    echo "Permissions of \$target changed to \$perm"
else
    echo "File/Directory not found!"
fi`
    }
  ];

  const [copiedIndex, setCopiedIndex] = useState(null);

  const handleCopy = async (index, text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 1500);
    } catch (err) {
      console.error("Copy failed:", err);
    }
  };

  return (
    <div className="page-container">
      <div className="gallery-grid">
        {codes.map((item, index) => (
          <div key={index} className="code-card">
            <div className="card-header">
              <div>
                <h2 className="card-title">{item.title}</h2>
                <p className="card-filename">{item.filename}</p>
              </div>
              <button
                onClick={() => handleCopy(index, item.code)}
                className="copy-button"
              >
                <Copy size={16} />
                {copiedIndex === index ? "Copied!" : "Copy"}
              </button>
            </div>
            <pre className="code-block">
              <code>{item.code}</code>
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
}
