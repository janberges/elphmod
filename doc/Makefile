.PHONY: clean html

html:
	sphinx-build -M html . .
	sed -i '/<img/ s|examples\S*/|_images/|g' html/examples/*.html

clean:
	rm -rf doctrees html
