function toggleDropdown() {
  const dropdown = document.getElementById("pacman-dropdown");
  dropdown.classList.toggle("hidden");

  const userResponse = confirm("Do you want to try the rain simulator?");
  if (userResponse) {
    window.open("https://kvnbbg.github.io/DynamicRainSimulator2.0/", "_blank");
  } else {
    const reason = prompt("Why don't you want to try the rain simulator?");
    // Store the response in a JSON object
    const responseInfo = {
      hour: new Date().getHours(),
      from: "User",
      answer: "No",
      response: reason,
    };
    // Convert the object into a string
    const responseJson = JSON.stringify(responseInfo);
    // Store the string in local storage
    localStorage.setItem("userResponse", responseJson);
  }

  if (!secondResponse) {
    const secondResponse = confirm(
      "Are you sure you don't want to try the new feature?",
    );
  }
}
window.onload = function() {
  const headers = document.querySelectorAll('h3');
  headers.forEach(header => {
    header.innerHTML = header.textContent.split('').map(letter => `<span>${letter}</span>`).join('');
  });
};

document.querySelector('.like-button').addEventListener('click', function() {
  this.textContent = this.textContent.includes('Liked') ? '❤️ Like' : '💔 Liked';
});