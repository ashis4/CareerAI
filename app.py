from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# ── Load model, tokenizer, label encoder ──
model = tf.keras.models.load_model('career_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

MAX_LEN = 50  # matches your training

# ── Keyword boost map (from your notebook) ──
keyword_map = {
    "Data Scientist": ["machine learning", "statistics", "pandas", "numpy"],
    "Data Analyst": ["sql", "excel", "power bi", "tableau"],
    "AI Engineer": ["deep learning", "nlp", "tensorflow", "pytorch"],
    "Machine Learning Engineer": ["ml pipeline", "deployment", "mlflow"],
    "Software Engineer": ["dsa", "algorithms", "oop", "coding"],
    "Java Developer": ["java", "spring", "hibernate"],
    "Frontend Developer": ["html", "css", "javascript", "react"],
    "Backend Developer": ["node", "django", "api", "database"],
    "Web Developer": ["mern", "full stack", "web app"],
    "Android Developer": ["android", "kotlin", "android studio"],
    "UI/UX Designer": ["figma", "wireframe", "prototype"],
    "Cloud Engineer": ["aws", "docker", "kubernetes"],
    "Cybersecurity Analyst": ["security", "hacking", "wireshark"],
    "Business Analyst": ["business analysis", "requirements"],
    "Financial Analyst": ["finance", "investment", "valuation"],
    "Digital Marketing Specialist": ["seo", "marketing", "ads"]
}

# ── Roadmaps ──
roadmaps = {

    "Backend Developer": """
    🖥️ Backend Developer Roadmap
1. Programming Language

Begin by learning a backend programming language such as Node.js, Python, or Java in depth. Focus on understanding how server-side logic works, how requests are received, processed, and responses are sent back to the client. Practice writing programs that handle data processing, file handling, and basic server creation. A strong command over one language will make it easier to learn frameworks and advanced backend concepts later.

2. Databases

Learn how to work with both relational databases like MySQL and PostgreSQL and NoSQL databases like MongoDB. Understand how data is stored, structured, and retrieved efficiently in real-world applications. Practice writing complex SQL queries, joins, indexing, and schema design for performance optimization. Knowing databases deeply is essential because backend systems heavily depend on data handling.

3. APIs

Move on to building RESTful APIs that enable communication between frontend and backend systems. Learn HTTP methods such as GET, POST, PUT, and DELETE, and understand how request-response cycles work. Practice building APIs that send and receive JSON data in a structured format. APIs are the backbone of modern applications and are used in almost every real-world system.

4. Authentication

Implement secure authentication systems using techniques like JWT (JSON Web Tokens) or session-based authentication. Learn how to securely store passwords using hashing algorithms like bcrypt and manage user sessions. Understand authorization concepts like role-based access control to restrict user permissions. Security is a critical part of backend development and must be handled carefully.

5. Frameworks

Use backend frameworks like Express.js (Node.js), Django (Python), or Spring Boot (Java) to simplify development. These frameworks provide built-in tools, structure, and best practices for building scalable applications. Learn how to create routes, handle middleware, and structure large applications properly. Frameworks are widely used in industry, so mastering one is essential.

6. Server & Deployment

Understand how servers work, including concepts like hosting, ports, and environment configurations. Learn how to deploy your backend applications using platforms like AWS, Render, or Heroku. Practice setting up environment variables and managing production vs development environments. Deployment helps you take your project from local machine to real-world usage.

7. Testing & Debugging

Learn how to test your backend applications using testing frameworks and tools. Practice debugging errors, handling edge cases, and improving code reliability. Write unit tests and integration tests to ensure your system works correctly under different scenarios. Clean and well-tested code is essential for maintaining production-level applications.

8. Projects

Build complete backend systems such as authentication systems, REST APIs, CRUD applications, or full-stack integrations. Apply everything you’ve learned in real-world scenarios and focus on solving practical problems. Make sure your projects are well-documented and hosted online. Strong projects are the key to getting noticed by recruiters.
    """,

    
    "Data Scientist": """
    🚀 Data Scientist Roadmap
1. Foundation

Start by building a strong foundation in Python along with key mathematical concepts such as statistics, probability, and linear algebra. These concepts help you understand how data behaves and how models make predictions. Spend time understanding distributions, mean, variance, and hypothesis testing in detail. A solid base here will make advanced topics much easier and more intuitive.

2. Data Preprocessing

Learn how to clean and prepare raw data before applying any models. Real-world data is often incomplete, noisy, and inconsistent, so handling missing values, duplicates, and outliers is essential. Practice techniques like encoding categorical variables and feature scaling. Good preprocessing directly improves model performance and reliability.

3. Exploratory Data Analysis (EDA)

Focus on analyzing datasets using visualization tools like Matplotlib and Seaborn. Understand patterns, trends, and relationships between different variables in the data. Use graphs like histograms, scatter plots, and heatmaps to gain insights. EDA helps you make informed decisions before moving to model building.

4. Machine Learning

Learn core machine learning algorithms such as linear regression, logistic regression, decision trees, and clustering methods. Understand how models are trained, tested, and evaluated using datasets. Focus on concepts like overfitting, underfitting, and bias-variance tradeoff. This is the stage where you start building predictive systems.

5. Feature Engineering

Improve your model’s performance by selecting and transforming the right features. Learn techniques like normalization, standardization, and creating new meaningful features from existing data. Understand dimensionality reduction methods like PCA. Strong feature engineering can significantly boost model accuracy.

6. Advanced Concepts

Work on advanced topics like hyperparameter tuning, cross-validation, and ensemble learning methods such as boosting and bagging. Learn how to optimize models for better performance and stability. Understand how different algorithms can be combined to improve results. This stage separates beginner-level understanding from professional-level skills.

7. Projects

Build real-world projects such as recommendation systems, prediction models, or analytics dashboards. Apply all your knowledge to solve practical problems using real datasets. Focus on writing clean code and documenting your work properly. Projects are essential to showcase your skills and understanding.

8. Deployment

Learn how to deploy your trained models using tools like Streamlit, Flask, or cloud platforms such as AWS. Understand how to create APIs and make your models accessible to users. Practice deploying end-to-end projects so they can be used in real-world applications. Deployment is what turns your work into a usable product.
    """,

    
    "Data Analyst": """
    📊 Data Analyst Roadmap
1. Foundation

Start by understanding basic concepts of data, statistics, and business thinking. Learn how data is collected, structured, and used for decision-making in organizations. Focus on concepts like mean, median, variance, and basic probability. A strong foundation helps you interpret data correctly and avoid misleading conclusions.

2. Excel Skills

Learn Microsoft Excel in depth as it is one of the most widely used tools in data analysis. Understand formulas, pivot tables, charts, and data cleaning techniques within Excel. Practice working with real datasets to analyze trends and patterns. Excel is often the first tool used in real business environments.

3. SQL

Master SQL to extract and manipulate data from databases efficiently. Learn commands like SELECT, JOIN, GROUP BY, and filtering techniques to work with large datasets. Practice writing complex queries to solve real-world business problems. SQL is a must-have skill for any data analyst role.

4. Data Cleaning

Learn how to clean messy data by handling missing values, duplicates, and inconsistencies. Understand how to transform raw data into a structured format suitable for analysis. Practice using tools like Excel or Python for cleaning tasks. Clean data ensures accurate and reliable insights.

5. Data Visualization

Use tools like Power BI, Tableau, or Python libraries to create interactive dashboards and visual reports. Learn how to represent data in a simple and meaningful way using charts and graphs. Focus on storytelling through visuals to communicate insights effectively. Visualization is key to decision-making.

6. Python for Analysis

Learn Python libraries like Pandas and NumPy to perform advanced data analysis. Automate repetitive tasks and handle large datasets efficiently. Use Python to clean, analyze, and visualize data beyond Excel capabilities. This skill helps you stand out from basic analysts.

7. Reporting & Communication

Develop the ability to present your findings clearly to stakeholders. Learn how to create reports and explain insights in simple business terms. Focus on storytelling with data rather than just showing numbers. Communication skills are crucial for a successful data analyst.

8. Projects

Build real-world projects such as dashboards, business reports, and data analysis case studies. Work with real datasets to solve practical problems and generate insights. Showcase your projects on GitHub or portfolio websites. Strong projects demonstrate your practical skills to employers.
    """,

    
    "AI Engineer": """
   🤖 AI Engineer Roadmap
1. Foundation

Begin with a strong understanding of Python programming along with mathematics such as linear algebra, probability, and statistics. These concepts are essential to understand how AI models work internally. Spend time learning vectors, matrices, and probability distributions in detail. A solid foundation will make it much easier to grasp complex AI algorithms later.

2. Machine Learning

Learn core machine learning techniques including supervised and unsupervised learning. Understand algorithms like linear regression, decision trees, and clustering methods. Focus on how models learn patterns from data and make predictions. Practice implementing these algorithms using libraries like Scikit-learn.

3. Deep Learning

Move on to deep learning concepts such as neural networks, CNNs (Convolutional Neural Networks), and RNNs (Recurrent Neural Networks). Understand how deep learning models handle complex data like images, audio, and sequences. Use frameworks like TensorFlow or PyTorch to build and train models. This step is crucial for advanced AI applications.

4. Natural Language Processing (NLP)

Learn how machines understand and process human language. Study techniques like tokenization, stemming, TF-IDF, and word embeddings. Build applications like chatbots, sentiment analysis systems, or text classifiers. NLP is widely used in real-world AI products.

5. Computer Vision

Work with image and video data using computer vision techniques. Learn image classification, object detection, and image segmentation. Use libraries like OpenCV and deep learning models for visual tasks. This domain is important for applications like facial recognition and autonomous systems.

6. Model Optimization

Focus on improving the performance and efficiency of your models. Learn techniques like hyperparameter tuning, regularization, and avoiding overfitting. Understand how to balance model accuracy and computational cost. Optimization is key to building reliable AI systems.

7. Projects

Build AI-based projects such as chatbots, recommendation systems, image classifiers, or voice assistants. Apply your knowledge to solve real-world problems using datasets. Ensure your projects are well-structured and documented. Projects demonstrate your ability to implement AI solutions.

8. Deployment

Learn how to deploy AI models into real applications using APIs and cloud platforms. Understand how to serve models using Flask, FastAPI, or cloud services like AWS. Make your AI systems accessible to users in real-time. Deployment is what turns your model into a usable product.
    """,

    
    "Machine Learning Engineer": """
   🧠 Machine Learning Engineer Roadmap
1. Foundation

Start by building a strong foundation in Python programming along with statistics and linear algebra. These concepts are essential to understand how machine learning models process data and make predictions. Focus on probability distributions, matrices, and basic statistical measures in depth. A solid base will help you implement and optimize models effectively.

2. Data Processing

Learn how to collect, clean, and preprocess data for machine learning tasks. Handle missing values, outliers, and inconsistent data formats carefully. Practice transforming raw data into structured formats suitable for model training. Good data processing is critical because the quality of data directly impacts model performance.

3. Machine Learning Algorithms

Understand and implement various machine learning algorithms such as regression, classification, and clustering. Learn how each algorithm works and when to use it. Practice building models using libraries like Scikit-learn. Strong knowledge of algorithms is essential for solving real-world problems.

4. Model Evaluation

Learn how to evaluate model performance using metrics like accuracy, precision, recall, and F1-score. Understand concepts like confusion matrix and ROC curves. Practice validating your models using techniques like cross-validation. Proper evaluation ensures your model performs well on unseen data.

5. Feature Engineering

Focus on improving input data by selecting and transforming relevant features. Learn techniques like encoding categorical variables, scaling, and dimensionality reduction. Create new features that enhance model performance. Effective feature engineering can significantly improve prediction accuracy.

6. Model Deployment

Learn how to deploy machine learning models using tools like Flask, FastAPI, or Streamlit. Understand how to create APIs that serve predictions in real-time. Practice integrating models into web applications. Deployment makes your models usable in real-world systems.

7. MLOps

Understand concepts of MLOps such as model versioning, monitoring, and automation of pipelines. Learn how to manage the lifecycle of machine learning models in production. Use tools that help in continuous integration and deployment of ML systems. MLOps is important for scaling ML solutions.

8. Projects

Build end-to-end machine learning projects that include data preprocessing, model training, evaluation, and deployment. Work on real-world datasets and focus on solving practical problems. Document your work clearly and showcase it in your portfolio. Strong projects demonstrate your complete ML pipeline understanding.
    """,

    
    "Software Engineer": """
 💻 Software Engineer Roadmap
1. Programming

Start by learning one or two programming languages such as Java, Python, or C++. Focus on understanding syntax, logic building, and writing clean, readable code. Practice solving basic problems to strengthen your problem-solving ability. A strong programming foundation is essential for all software development roles.

2. Data Structures & Algorithms

Learn important data structures like arrays, linked lists, stacks, queues, trees, and graphs. Understand algorithms for searching, sorting, and optimization problems. Practice solving problems on platforms like LeetCode or HackerRank. This knowledge is crucial for technical interviews and efficient coding.

3. Object-Oriented Programming

Understand core OOP concepts such as classes, objects, inheritance, polymorphism, and encapsulation. Learn how to design modular and reusable code using these principles. OOP helps in building scalable and maintainable software systems. It is widely used in real-world applications.

4. Development Practices

Learn software development methodologies like Agile and Scrum. Understand how to write clean, maintainable, and well-documented code. Follow coding standards and best practices used in the industry. This helps in working efficiently in team environments.

5. Version Control

Use Git and GitHub to manage your code and track changes over time. Learn how to create branches, merge code, and collaborate with other developers. Version control is essential for teamwork and project management. It also helps in maintaining project history.

6. System Design

Learn how to design scalable and efficient systems for real-world applications. Understand concepts like load balancing, caching, database design, and microservices. Practice designing systems like chat apps or e-commerce platforms. System design is important for senior-level roles.

7. Testing

Understand different types of testing such as unit testing, integration testing, and debugging. Learn how to identify and fix errors efficiently. Writing tested and reliable code ensures better software quality. Testing is a key part of professional development workflows.

8. Projects

Build real-world applications such as web apps, APIs, or system-based software. Apply all your knowledge to solve practical problems. Focus on writing clean code and proper documentation. Strong projects help you stand out during placements and interviews.
    """,

    
    "Java Developer": """
    ☕ Java Developer Roadmap
1. Core Java

Begin by learning core Java concepts such as syntax, data types, loops, and control statements. Focus on understanding object-oriented programming principles like classes, objects, inheritance, and polymorphism. Practice writing small programs to strengthen your logic and coding skills. A strong foundation in core Java is essential before moving to advanced topics.

2. Advanced Java

Move on to advanced Java topics like JDBC, Servlets, and JSP, which are used for building web applications. Learn how Java interacts with databases and how server-side processing works. Understand the lifecycle of web applications and request handling. These concepts are important for developing dynamic applications.

3. Frameworks

Learn popular Java frameworks such as Spring and Spring Boot to build scalable and efficient applications. Understand dependency injection, REST API development, and application configuration. Frameworks simplify development and are widely used in the industry. Mastering Spring Boot is especially valuable for backend development.

4. Database Integration

Understand how to connect Java applications with databases using JDBC or ORM tools like Hibernate. Learn how to perform CRUD operations and manage database transactions. Practice writing efficient queries and optimizing database performance. Database integration is crucial for real-world applications.

5. API Development

Learn how to design and develop RESTful APIs using Spring Boot. Understand how to handle HTTP requests, responses, and JSON data. Practice building APIs that can be consumed by frontend or mobile applications. APIs are essential for communication between different systems.

6. Build Tools

Use tools like Maven or Gradle to manage project dependencies and automate the build process. Learn how to structure projects and handle external libraries efficiently. These tools improve productivity and maintain consistency in development. They are commonly used in professional environments.

7. Testing & Debugging

Learn testing frameworks like JUnit to write unit tests for your applications. Practice debugging errors and improving code performance. Ensure your code is reliable and works correctly under different scenarios. Testing is important for maintaining high-quality software.

8. Projects

Build real-world Java applications such as management systems, REST APIs, or enterprise-level software. Apply all your knowledge in practical scenarios and focus on solving real problems. Document your projects and host them on GitHub. Strong projects will enhance your portfolio and job opportunities.
    """,

    
    "Frontend Developer": """
  🎨 Frontend Developer Roadmap
1. Basics

Start by learning HTML, CSS, and JavaScript, which are the core building blocks of web development. Understand how HTML structures a webpage, how CSS is used for styling, and how JavaScript adds interactivity. Practice creating simple static websites to strengthen your fundamentals. A strong base in these technologies is essential before moving to advanced concepts.

2. Styling

Focus on advanced CSS techniques such as Flexbox and Grid to create modern layouts. Learn responsive design so your websites work smoothly across mobile, tablet, and desktop devices. Understand media queries and mobile-first design principles. Good styling skills help you build visually appealing and user-friendly interfaces.

3. JavaScript Advanced

Deep dive into JavaScript concepts like DOM manipulation, event handling, promises, and async/await. Learn ES6+ features such as arrow functions, destructuring, and modules. These concepts help in building dynamic and interactive web applications. Writing clean and efficient JavaScript code is very important.

4. Frameworks

Learn modern frontend frameworks like React, Angular, or Vue to build scalable applications. Understand component-based architecture and how to manage state and routing. Frameworks help you organize code and reuse components efficiently. They are widely used in industry-level applications.

5. APIs

Learn how to fetch and display data from backend services using APIs. Understand how to work with JSON data and asynchronous operations. Practice integrating APIs into your applications to make them dynamic. This is essential for building real-world web applications.

6. Version Control

Use Git and GitHub to manage your code and track changes over time. Learn how to create branches, merge code, and collaborate with other developers. Version control helps maintain project history and teamwork efficiency. It is an essential skill in professional development.

7. Performance Optimization

Focus on improving the speed and performance of your web applications. Learn techniques like lazy loading, code splitting, and image optimization. Reduce unnecessary re-renders and improve loading time. Better performance leads to improved user experience.

8. Projects

Build real-world projects such as portfolios, dashboards, or e-commerce frontends. Apply your knowledge to create responsive and interactive web applications. Focus on clean design and usability. Strong projects will help you stand out in job applications.
    """,

    
    "Web Developer": """
   🌐 Web Developer Roadmap
1. Basics

Start by learning HTML, CSS, and JavaScript to understand how websites are built from scratch. Focus on creating structured web pages using semantic HTML and styling them effectively with CSS. Use JavaScript to add interactivity and dynamic behavior. Building small static websites at this stage will strengthen your foundation.

2. Frontend Development

Learn advanced frontend concepts such as responsive design, modern CSS techniques, and JavaScript frameworks like React. Understand how to build user-friendly interfaces and manage UI components efficiently. Focus on improving user experience and design consistency. This step helps you create visually appealing web applications.

3. Backend Development

Learn server-side programming using technologies like Node.js, Python, or Java. Understand how to handle requests, process data, and interact with databases. Build APIs that connect frontend and backend systems. This step is essential for creating complete web applications.

4. Databases

Learn how to work with databases such as MySQL, PostgreSQL, or MongoDB. Understand how to store, retrieve, and manage data efficiently. Practice designing database schemas and writing queries. Databases are crucial for handling application data.

5. Full Stack Integration

Learn how to connect frontend and backend using APIs. Understand how data flows between client and server in real-world applications. Practice building full-stack applications that combine both frontend and backend. This step helps you become a complete web developer.

6. Version Control

Use Git and GitHub to manage your projects and track changes. Learn how to collaborate with others using branches and pull requests. Version control helps maintain code quality and project organization. It is essential for team-based development.

7. Deployment

Learn how to deploy web applications using platforms like Netlify, Vercel, or AWS. Understand hosting, domain setup, and environment configuration. Make your applications accessible online for real users. Deployment is a key step in making your work live.

8. Projects

Build complete web applications such as blogs, e-commerce websites, or management systems. Apply all your skills to solve real-world problems. Focus on functionality, design, and performance. Strong projects will enhance your portfolio and career opportunities.
    """,

    
    "Android Developer": """
    📱 Android Developer Roadmap
1. Programming

Start by learning Java or Kotlin, which are the primary languages used for Android development. Focus on understanding syntax, object-oriented programming concepts, and basic logic building. Practice writing simple programs to strengthen your coding skills. A strong programming base is essential before moving into Android-specific development.

2. Android Fundamentals

Learn the core components of Android such as Activities, Fragments, Intents, and layouts. Understand how the Android lifecycle works and how different components interact. Practice building simple apps to understand navigation and UI flow. This step forms the backbone of Android app development.

3. UI Design

Focus on designing user-friendly and visually appealing mobile interfaces. Learn XML for designing layouts and understand Material Design principles. Ensure your apps are responsive across different screen sizes and devices. Good UI design improves user experience significantly.

4. Data Storage

Learn how to store and manage data using SQLite, Room Database, or Firebase. Understand how to save user data locally and retrieve it efficiently. Practice handling structured data within your applications. Data management is important for building functional apps.

5. APIs Integration

Learn how to connect your app with external services using APIs. Understand how to send requests, receive responses, and handle JSON data. Practice integrating features like login systems, maps, or payment gateways. APIs make your app dynamic and feature-rich.

6. Testing & Debugging

Learn how to test your applications and fix bugs efficiently. Use debugging tools to identify issues in your code. Ensure your app runs smoothly under different conditions and devices. Testing improves app quality and user satisfaction.

7. Deployment

Understand how to prepare your app for release on the Google Play Store. Learn about app signing, versioning, and publishing guidelines. Optimize your app for performance and security before release. Deployment makes your app available to real users.

8. Projects

Build real-world Android applications such as task managers, chat apps, or utility tools. Apply your knowledge to solve practical problems and improve app functionality. Showcase your apps on GitHub or Play Store. Strong projects help you stand out in job opportunities.
    """,

    
    "UI/UX Designer": """
   🎯 UI/UX Designer Roadmap
1. Design Fundamentals

Start by learning the basic principles of design such as color theory, typography, spacing, and layout. Understand how visual hierarchy and alignment affect user perception. Study how good design improves usability and user satisfaction. A strong foundation in design principles is essential for creating effective interfaces.

2. Design Tools

Learn industry-standard tools like Figma, Adobe XD, or Sketch for designing user interfaces. Understand how to create layouts, components, and reusable design systems. Practice building designs for web and mobile applications. These tools are essential for professional UI/UX work.

3. User Research

Understand user needs by conducting research through surveys, interviews, or observations. Learn how to create user personas and define user journeys. Focus on identifying pain points and user behavior. Good research helps in designing user-centered solutions.

4. Wireframing

Create low-fidelity wireframes to plan the structure and layout of applications. Focus on functionality and user flow rather than visual design. Use tools to sketch ideas quickly and iterate on them. Wireframing helps in visualizing the product before development.

5. Prototyping

Build interactive prototypes that simulate real user interactions. Learn how to connect screens and create clickable designs. Prototypes help stakeholders understand how the final product will work. This step bridges the gap between design and development.

6. Usability Testing

Test your designs with real users to identify issues and improve usability. Collect feedback and refine your designs based on user behavior. Focus on making the interface intuitive and easy to use. Testing ensures a better user experience.

7. Design Systems

Learn how to create and maintain consistent design systems with reusable components. Ensure consistency in colors, typography, and UI elements across projects. Design systems improve efficiency and scalability. They are widely used in large applications.

8. Projects

Build real-world design projects such as app interfaces, website layouts, or redesign case studies. Showcase your work in a portfolio with proper explanations. Focus on solving real user problems through design. Strong projects are essential for getting UI/UX roles.
    """,

    
    "Cloud Engineer": """
   ☁️ Cloud Engineer Roadmap
1. Fundamentals

Start by understanding the basics of cloud computing, including concepts like virtualization, networking, and storage. Learn how cloud platforms provide on-demand resources over the internet. Understand service models like IaaS, PaaS, and SaaS. A strong foundation helps you grasp how cloud systems operate in real-world environments.

2. Cloud Platforms

Learn major cloud platforms such as AWS, Microsoft Azure, or Google Cloud Platform. Understand their core services like compute, storage, and networking. Practice using these platforms through hands-on labs and small projects. Cloud platform knowledge is essential for working in industry.

3. Operating Systems

Gain knowledge of operating systems, especially Linux, as it is widely used in cloud environments. Learn basic commands, file systems, and process management. Understand how servers run applications and handle resources. This helps in managing cloud-based systems effectively.

4. Networking

Understand networking concepts such as IP addressing, DNS, load balancing, and firewalls. Learn how data flows between systems and how networks are secured. Practice configuring virtual networks in cloud platforms. Networking is a key component of cloud infrastructure.

5. DevOps Tools

Learn tools like Docker, Kubernetes, Jenkins, and CI/CD pipelines. Understand how to automate deployment and manage containerized applications. DevOps practices improve efficiency and scalability. These tools are widely used in modern cloud environments.

6. Security

Understand cloud security concepts such as identity and access management (IAM), encryption, and secure configurations. Learn how to protect cloud resources from threats and vulnerabilities. Security is a critical responsibility in cloud engineering.

7. Deployment & Monitoring

Learn how to deploy applications on cloud platforms and monitor their performance. Use tools to track system health, logs, and usage metrics. Understand scaling and load management. Monitoring ensures system reliability and performance.

8. Projects

Build real-world cloud projects such as deploying web applications, setting up cloud infrastructure, or creating CI/CD pipelines. Apply your knowledge to practical scenarios. Showcase your projects to demonstrate your cloud skills. Strong projects are essential for cloud engineering roles.
    """,

    
    "Cybersecurity Analyst": """
  🔐 Cybersecurity Analyst Roadmap
1. Fundamentals

Start by understanding the basics of cybersecurity, including concepts like confidentiality, integrity, and availability (CIA triad). Learn how systems, networks, and data can be vulnerable to attacks. Study common threats such as malware, phishing, and ransomware. A strong foundation helps you understand how security systems are designed and protected.

2. Networking

Learn networking concepts such as TCP/IP, DNS, firewalls, and protocols. Understand how data flows across networks and how attackers exploit vulnerabilities. Practice analyzing network traffic and identifying suspicious activities. Networking knowledge is essential for detecting and preventing cyber attacks.

3. Operating Systems

Gain deep knowledge of operating systems like Windows and Linux. Understand system architecture, file systems, and permissions. Learn how attackers exploit OS vulnerabilities and how to secure them. This helps in identifying threats at the system level.

4. Security Tools

Learn to use cybersecurity tools like Wireshark, Metasploit, Nmap, and antivirus systems. Understand how these tools help in scanning, monitoring, and testing system security. Practice hands-on labs to simulate attacks and defenses. Tools are essential for real-world cybersecurity work.

5. Ethical Hacking

Learn ethical hacking techniques to understand how attackers think and operate. Practice penetration testing to identify vulnerabilities in systems. Understand different types of attacks and how to defend against them. Ethical hacking helps strengthen system security.

6. Risk Management

Understand how to assess risks and implement security policies. Learn about compliance standards and frameworks used in organizations. Focus on minimizing threats and protecting sensitive data. Risk management is important for maintaining secure systems.

7. Incident Response

Learn how to detect, respond to, and recover from security incidents. Understand how to analyze logs, investigate breaches, and take corrective actions. Practice creating incident response plans. Quick and effective response reduces damage from attacks.

8. Projects

Work on cybersecurity projects such as vulnerability assessment, penetration testing reports, or network security setups. Apply your knowledge to real-world scenarios. Document your findings and solutions clearly. Strong projects help demonstrate your practical skills.
    """,

    
    "Business Analyst": """
   📈 Business Analyst Roadmap
1. Business Fundamentals

Start by understanding how businesses operate, including processes, workflows, and organizational structure. Learn basic concepts of management, finance, and operations. Focus on how companies make decisions and solve problems. A strong business foundation helps you align data insights with business goals.

2. Analytical Thinking

Develop strong analytical and problem-solving skills to interpret data and identify patterns. Learn how to break down complex problems into smaller parts. Focus on logical thinking and decision-making based on data. These skills are essential for providing meaningful business insights.

3. Excel & Tools

Learn tools like Microsoft Excel for data analysis, including formulas, pivot tables, and charts. Understand how to manipulate and analyze datasets effectively. Excel is widely used in business environments for reporting and analysis. Mastering it will give you a strong advantage.

4. SQL

Learn SQL to extract and analyze data from databases. Understand how to write queries to retrieve useful business insights. Practice working with real datasets and solving business problems using SQL. This skill is essential for handling structured data.

5. Data Visualization

Use tools like Power BI, Tableau, or Excel dashboards to present data visually. Learn how to create meaningful charts and reports for stakeholders. Focus on storytelling with data to communicate insights clearly. Visualization helps in better decision-making.

6. Requirement Gathering

Learn how to gather and document business requirements from stakeholders. Understand techniques like interviews, surveys, and use case analysis. Focus on translating business needs into technical solutions. This is a key responsibility of a business analyst.

7. Communication Skills

Develop strong communication and presentation skills to explain insights and solutions. Learn how to interact with both technical and non-technical teams. Clear communication ensures that everyone understands the business objectives. It is crucial for successful project execution.

8. Projects

Work on real-world case studies and business analysis projects. Create reports, dashboards, and requirement documents. Apply your knowledge to solve business problems effectively. Strong projects will help you build a solid portfolio.
    """,

    
    "Financial Analyst": """
   💰 Financial Analyst Roadmap
1. Financial Fundamentals

Start by learning the basics of finance such as financial statements, accounting principles, and key financial concepts. Understand how income statements, balance sheets, and cash flow statements work together. Focus on interpreting financial data to assess a company’s performance. A strong foundation in finance is essential for accurate analysis.

2. Accounting Knowledge

Develop a solid understanding of accounting principles like assets, liabilities, revenue, and expenses. Learn how transactions are recorded and how financial reports are generated. Practice analyzing company financial reports. Accounting knowledge helps you understand the financial health of an organization.

3. Excel & Financial Tools

Learn advanced Excel skills including formulas, financial modeling, pivot tables, and data analysis tools. Practice building financial models and forecasting future performance. Excel is one of the most important tools for financial analysts. Mastering it will significantly improve your efficiency.

4. Financial Analysis

Learn techniques like ratio analysis, trend analysis, and comparative analysis. Understand how to evaluate profitability, liquidity, and solvency of a company. Practice analyzing real company data. Financial analysis helps in making informed investment and business decisions.

5. Valuation Techniques

Understand methods like Discounted Cash Flow (DCF), comparable company analysis, and asset-based valuation. Learn how to estimate the value of a company or investment. Practice applying these methods to real-world scenarios. Valuation is a core skill for financial analysts.

6. Risk Management

Learn how to identify and manage financial risks. Understand concepts like diversification, hedging, and risk-return tradeoff. Analyze how external factors impact financial performance. Risk management helps in making safer financial decisions.

7. Reporting & Communication

Develop skills to present financial insights clearly through reports and presentations. Learn how to explain complex financial data in simple terms. Focus on storytelling with numbers for stakeholders. Communication is key in financial decision-making.

8. Projects

Work on financial analysis projects such as company valuation, stock analysis, or financial forecasting. Use real-world data to practice your skills. Document your findings and present them clearly. Strong projects will enhance your portfolio and job opportunities.
    """,

    "Digital Marketing Specialist": """
  📢 Digital Marketing Specialist Roadmap
1. Marketing Fundamentals

Start by understanding the core concepts of marketing such as target audience, branding, and customer behavior. Learn how businesses promote products and services across different channels. Focus on concepts like the marketing funnel, segmentation, and positioning. A strong foundation helps you create effective marketing strategies.

2. SEO (Search Engine Optimization)

Learn how search engines work and how to optimize websites to rank higher in search results. Understand keyword research, on-page SEO, and off-page SEO techniques. Practice improving website visibility and organic traffic. SEO is essential for long-term digital growth.

3. Social Media Marketing

Learn how to promote brands using platforms like Instagram, Facebook, LinkedIn, and Twitter. Understand content creation, audience engagement, and campaign strategies. Analyze trends and create posts that attract users. Social media plays a major role in digital marketing.

4. Content Marketing

Focus on creating valuable and engaging content such as blogs, videos, and social posts. Learn how to write content that attracts and retains customers. Understand storytelling and brand voice. Good content helps build trust and drive traffic.

5. Paid Advertising

Learn how to run paid campaigns using platforms like Google Ads and social media ads. Understand targeting, budgeting, and performance tracking. Practice creating effective ad campaigns to generate leads and conversions. Paid ads help in quick business growth.

6. Analytics & Tools

Learn tools like Google Analytics, Search Console, and marketing dashboards. Understand how to track website traffic, user behavior, and campaign performance. Analyze data to improve marketing strategies. Data-driven decisions are key in digital marketing.

7. Email Marketing

Learn how to create and manage email campaigns for customer engagement. Understand list building, personalization, and automation. Practice writing effective email content that drives action. Email marketing is important for retention and conversions.

8. Projects

Work on real-world marketing campaigns such as social media pages, SEO projects, or ad campaigns. Apply your knowledge to promote a product or service. Track performance and optimize results. Strong projects help you build a practical portfolio.
    """
}


def clean_text(text):
    text = text.lower()
    text = text.replace(",", " ")
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_career(text):
    text_clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([text_clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    pred = model.predict(padded)[0]

    # Keyword boost
    for role, keywords in keyword_map.items():
        for word in keywords:
            if word in text_clean:
                idx = list(label_encoder.classes_).index(role)
                pred[idx] += 0.8

    pred = pred / pred.sum()
    top3 = np.argsort(pred)[-3:][::-1]

    return [{"role": label_encoder.classes_[i], "score": round(float(pred[i]) * 100, 2)} for i in top3]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    skills = data.get('skills', '').strip()

    if not skills:
        return jsonify({'error': 'No skills provided'}), 400

    results = predict_career(skills)
    top_role = results[0]['role']

    return jsonify({
        'prediction': top_role,
        'top3': results,
        'roadmaps': roadmaps
    })


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
    
